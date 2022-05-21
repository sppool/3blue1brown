"""
key 'python COVID-19.py' to run

start from here "3blue1brown"
"https://www.youtube.com/watch?v=gxAaO2rsdIs"

status  狀態值解釋
0         : 健康的人
1   ~ 50  : 受感染但'沒有'傳染力 (正常更新時+1)
51  ~ 500 : 受感染且有傳染力 (正常更新時+1)
501 ~ 600 : 死亡or隔離(無傳染力), 單純是畫面顯示需要 (正常更新時+1)
601       : 不再更新狀態

r (float) (約0.001~0.002) 接觸距離(半徑)
n (int) 點位數量 (1000) 太少可以增加 r 提高接觸機會

初始感染率 : 1 - ((1 - r**2 * np.pi)**(n - 1))**450
* (n - 1) 是健康人人數
* 450是存活的步數
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from sklearn.metrics.pairwise import manhattan_distances  # 曼哈頓距離可以省大量計算


def update_coor_mov_status(coor, mov, status):
    status[np.bitwise_and(status > 0, status <= 600)] += 1  # 狀態更新
    g = coor[status == 0]  # 健康人座標
    r = coor[np.bitwise_and(status > 50, status <= 500)]  # 病人座標
    if len(g) and len(r):  # 當病人或是健康人數為0 不在更新
        d_arr = manhattan_distances(r, g)  # 計算病人和健康人曼哈頓距離(array)
        # 距離太近就傳染 status轉換為1(病人)
        g2r = np.any(d_arr < d, 0).astype(np.uint8)  # bool to int
        status[status == 0] += g2r  # 傳染更新

    mov[np.bitwise_and(status > 500, status <= 600)] *= 0.975  # 死亡移動變慢
    coor[status <= 600] = coor[status <= 600] + mov[status <= 600]  # 更新位置
    # 碰撞轉換
    upper = coor > 1
    lower = coor < 0
    coor[upper] = 2 - coor[upper]
    coor[lower] = -coor[lower]
    # 碰撞向量轉換
    mov[upper] = -mov[upper]
    mov[lower] = -mov[lower]

    return coor, mov, status


n = 1000  # 模擬人數(太多很爽 但是會lag 吃光記憶體會當機 請緩慢增加)
r = 0.0015  # 距離設定 微調影響很顯著 約0.001~0.002
d = r * (np.pi / 2)**0.5  # 轉換成曼哈頓距離(讓面積相等) 可以減少計算 且已經足夠
# 初始機率 0.75~0.95 (r 0.001 ~ 0.0015)有明顯差距
print(f'p: {1 - ((1 - r**2 * np.pi)**(n - 1))**450}')
time.sleep(1)

# 狀態設定 初始0號病人 (0:健康人, 1~500:病人, 500~:凋零不再傳染)
status = np.zeros(n).astype(np.uint16)
status[0] = 1  # 零號COVID-19病患
angle = np.random.uniform(0, 2 * np.pi, n)  # 移動角度
speed = np.random.uniform(0.3, 1, n)  # 移動速度(可以直接設 1 移動速度相同)
comp = np.exp(angle * 1j) * speed  # 移動向量
step = 400  # 解析度(越大步數越小 直接影響流暢度)
mov = np.vstack((comp.real, comp.imag)).transpose(1, 0) / step
mov = mov.astype(np.float32)  # 節省資源 設定 np.float16 會不夠
coor = np.random.uniform(0, 1, size=(n, 2))  # 位置座標
coor = coor.astype(np.float32)  # 節省資源

fig_size = 7  # init圖形大小
fig, ax = plt.subplots(figsize=(fig_size, fig_size))
p4, = ax.plot('', '', 'o', ms=3, c='#404040', alpha=0.5)  # 最低圖層
p2, = ax.plot('', '', 'o', ms=15, c='#ffa000', alpha=0.7)
p1, = ax.plot('', '', 'og', ms=5, alpha=0.7)
p3, = ax.plot('', '', 'or', ms=5, alpha=0.7)


def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def update(frame):
    global coor, mov, status
    g_coor = coor[status == 0]  # 健康人座標
    y_coor = coor[np.bitwise_and(status > 0, status <= 50)]  # 潛伏期病人座標
    r_coor = coor[np.bitwise_and(status > 0, status <= 500)]  # 病人座標
    k_coor = coor[np.bitwise_and(status > 500, status <= 600)]  # 死亡後消失
    p1.set_data(g_coor[:, 0], g_coor[:, 1])
    p2.set_data(y_coor[:, 0], y_coor[:, 1])
    p3.set_data(r_coor[:, 0], r_coor[:, 1])
    p4.set_data(k_coor[:, 0], k_coor[:, 1])
    print(f'{len(g_coor)}, {len(r_coor)}')

    coor, mov, status = update_coor_mov_status(coor, mov, status)


ani = FuncAnimation(fig=fig, func=update, frames=1,
                    init_func=init, interval=20, blit=False)  # fps=50
plt.show()
