# RUN: llvm-mc --triple=riscv64 -mattr=+v,+experimental-zvabd < %s --show-encoding 2>&1 \
# RUN:   | FileCheck --check-prefix=ALIAS %s
# RUN: llvm-mc --triple=riscv64 -mattr=+v,+experimental-zvabd --M no-aliases < %s --show-encoding 2>&1 \
# RUN:   | FileCheck --check-prefix=NO-ALIAS %s

# ALIAS:    vabs.v v2, v1 # encoding: [0x57,0x61,0x10,0x46]
# NO-ALIAS: vabd.vx v2, v1, zero # encoding: [0x57,0x61,0x10,0x46]
vabs.v v2, v1

# ALIAS:    vabs.v v2, v1, v0.t # encoding: [0x57,0x61,0x10,0x44]
# NO-ALIAS: vabd.vx v2, v1, zero, v0.t # encoding: [0x57,0x61,0x10,0x44]
vabs.v v2, v1, v0.t
