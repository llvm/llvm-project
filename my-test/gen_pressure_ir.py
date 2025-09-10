#!/usr/bin/env python3
import sys

def main():
    N = 2000  # 預設同時活躍數量
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])

    print("; Auto-generated high-pressure IR")
    print("; N =", N)
    print('declare i64 @opaque(i64)')
    print('declare void @sink(...)')
    print('')
    # 用 optnone+noinline 把最佳化關掉、避免內聯
    print('define i64 @massive(i64 %seed) optnone noinline {')
    print('entry:')

    # 逐一建立加法 + 黑箱呼叫，得到 v0..v(N-1)
    for i in range(N):
        print(f'  %add{i} = add i64 %seed, {i+1}')
        print(f'  %v{i} = call i64 @opaque(i64 %add{i})')

    # 把所有 vX 一口氣傳給 sink，確保「同時活躍」
    # vararg 呼叫要在 call 站位型別上標註第一個參數型別
    print('  call void (i64, ...) @sink(' + ', '.join([f'i64 %v{i}' for i in range(N)]) + ')')

    # 隨便回傳一個值，避免 void
    print('  ret i64 %v0')
    print('}')
    print('')
    print('; 建議用法：')
    print(';   python3 gen_pressure_ir.py 2000 > massive.ll')
    print(';   llc massive.ll -O2 -regalloc=greedy -o out.s')
    print(';   # 或換自己的 allocator 比較：-regalloc=segmenttree')

if __name__ == "__main__":
    main()
