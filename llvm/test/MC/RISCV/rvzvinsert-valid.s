# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zvinsert -show-encoding %s | \
# RUN:     llvm-objdump -d --mattr=+zvinsert -M no-aliases -M numeric - | \
# RUN:     FileCheck -check-prefixes=CHECK %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zvinsert -show-encoding %s | \
# RUN:     llvm-objdump -d --mattr=+zvinsert -M no-aliases -M numeric - | \
# RUN:     FileCheck -check-prefixes=CHECK %s

# CHECK: 501fb1d7      vinserti.s.x    v3, x1, 0x1f
vinserti.s.x v3, x1, 31

# CHECK: 5013c1d7      vinsert.s.x     v3, x1, (x7)
vinsert.s.x v3, x1, (x7)

# CHECK: 547fb1d7      vextracti.x.s   x3, v7, 0x1f
vextracti.x.s x3, v7, 31

# CHECK: 5453c1d7      vextract.x.s    x3, v5, (x7)
vextract.x.s x3, v5, (x7)