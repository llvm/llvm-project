# gen_high_pressure_ir.py
import sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024

print('declare void @llvm.trap()')  # 只是避免 LTO/IPO 想太多；其實用不到
print('define void @hp(i64 %x) noinline optnone {')
print('entry:')

# 1) 先生成 N 個「一定需要放在暫存器」的定義：
#    用一條條有輸出的 inline asm： "=&r" 表示早期破壞，避免與其他值合併。
#    模板用最簡單的 mov 把輸入複製到輸出，確保真的產生虛擬暫存器定義。
for i in range(N):
    print(f'  %t{i} = call i64 asm sideeffect "mov $0, $1", "=&r,r"(i64 %x)')

# 2) 最後用一條「只讀入、不輸出」的 inline asm，把所有 %t[i] 當作 "r" 輸入一次性吃掉。
#    加上 sideeffect 與 ~{memory}，抑制刪除與重排，強迫在同一點同時活著。
constraints = ",".join(["r"] * N)
operands = ", ".join([f"i64 %t{i}" for i in range(N)])
print(f'  call void asm sideeffect "", "{constraints},~{{memory}}"({operands})')

print('  ret void')
print('}')
