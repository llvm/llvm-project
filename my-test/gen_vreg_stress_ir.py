# gen_vreg_stress_ir.py
N = 1024

print('; ModuleID = "vreg-stress"')
print('source_filename = "vreg-stress.ll"')
print('target triple = "x86_64-pc-linux-gnu"\n')

print('define i32 @f(i32 %x) {')
print('entry:')

for i in range(N):
    print(f'  %v{i} = add i32 %x, {i}')
print('  br label %use\n')

print('use:')
if N == 1:
    print('  ret i32 %v0')
else:
    print('  %acc0 = add i32 %v0, %v1')
    for i in range(2, N+1):   # ★ 到 N（含）讓最後成為 acc(N-1)
        src = f'%acc{i-2}' if i-2 >= 0 else '%v0'
        rhs = f'%v{i}' if i < N else '0'
        print(f'  %acc{i-1} = add i32 {src}, {rhs}')
    print(f'  ret i32 %acc{N-1}')
print('}')
