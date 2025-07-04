# RUN: llvm-mc --triple=riscv64 -mattr=+v < %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc --triple=riscv64 -mattr=+v < %s -filetype=obj -o - \
# RUN:   | llvm-objdump -d --mattr=+v - \
# RUN:   | FileCheck %s --check-prefix=OBJ


.set absdef, 1

## simm5_plus1
# ASM: vmsgtu.vi v3, v4, 0, v0.t
# OBJ: vmsgtu.vi v3, v4, 0x0, v0.t
vmsgeu.vi v3, v4, absdef, v0.t

## simm5
# ASM: vadd.vi v4, v5, 1, v0.t
# OBJ: vadd.vi v4, v5, 0x1, v0.t
vadd.vi v4, v5, absdef, v0.t

