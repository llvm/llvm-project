# RUN: llvm-mc -triple riscv64 -mattr=+c -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -show-encoding \
# RUN:   -M no-aliases < %s | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# Tests compressed instructions available in rv64 and not in rv32.

# CHECK-BYTES: 7fe0
# CHECK-ALIAS: ld s0, 248(a5)
# CHECK-INST: c.ld s0, 248(a5)
# CHECK: # encoding: [0xe0,0x7f]
ld s0, 248(a5)

# CHECK-BYTES: e3a0
# CHECK-ALIAS: sd s0, 64(a5)
# CHECK-INST: c.sd s0, 64(a5)
# CHECK: # encoding: [0xa0,0xe3]
sd s0, 64(a5)

# CHECK-BYTES: 227d
# CHEACK-ALIAS: addiw tp, tp, 31
# CHECK-INST: c.addiw  tp, 31
# CHECK: # encoding: [0x7d,0x22]
addiw tp, tp, 31

# CHECK-BYTES: 9c1d
# CHEACK-ALIAS: subw s0, s0, a5
# CHECK-INST: c.subw s0, a5
# CHECK: # encoding:  [0x1d,0x9c]
subw s0, s0, a5

# CHECK-BYTES: 9c3d
# CHECK-ALIAS: addw s0, s0, a5
# CHECK-INST: c.addw s0, a5
# CHECK: # encoding: [0x3d,0x9c]
addw s0, s0, a5

# CHECK-BYTES: 9c3d
# CHECK-ALIAS: addw s0, s0, a5
# CHECK-INST: c.addw s0, a5
# CHECK: # encoding: [0x3d,0x9c]
addw s0, a5, s0

# CHECK-BYTES: 70ee
# CHECK-ALIAS: ld ra, 248(sp)
# CHECK-INST: c.ldsp ra, 248(sp)
# CHECK: # encoding:  [0xee,0x70]
ld ra, 248(sp)

# CHECK-BYTES: e0a2
# CHECK-ALIAS: sd s0, 64(sp)
# CHECK-INST: c.sdsp s0, 64(sp)
# CHECK: # encoding: [0xa2,0xe0]
sd s0, 64(sp)
