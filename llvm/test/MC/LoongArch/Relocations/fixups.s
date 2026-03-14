# RUN: llvm-mc --triple=loongarch64 %s --show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-FIXUP %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s \
# RUN:     | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=CHECK-REL %s

## Checks that fixups that can be resolved within the same object file are
## applied correctly.

.LBB0:
lu12i.w $t1, %abs_hi20(val)
# CHECK-FIXUP: fixup A - offset: 0, value: %abs_hi20(val), kind: fixup_loongarch_abs_hi20
# CHECK-INSTR: lu12i.w $t1, 74565

ori $t1, $t1, %abs_lo12(val)
# CHECK-FIXUP: fixup A - offset: 0, value: %abs_lo12(val), kind: fixup_loongarch_abs_lo12
# CHECK-INSTR: ori $t1, $t1, 1656

b .LBB0
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_loongarch_b26
# CHECK-INSTR: b -8
b .LBB2
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB2, kind: fixup_loongarch_b26
# CHECK-INSTR: b 331004
beq $a0, $a1, .LBB0
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_loongarch_b16
# CHECK-INSTR: beq $a0, $a1, -16
blt $a0, $a1, .LBB1
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB1, kind: fixup_loongarch_b16
# CHECK-INSTR: blt $a0, $a1, 1116
beqz $a0, .LBB0
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_loongarch_b21
# CHECK-INSTR: beqz $a0, -24
bnez $a0, .LBB1
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB1, kind: fixup_loongarch_b21
# CHECK-INSTR: bnez $a0, 1108

.fill 1104

.LBB1:

.fill 329876
nop
.LBB2:

.set val, 0x12345678

# CHECK-REL-NOT: R_LARCH

## Testing the function call offset could resolved by assembler
## when the function and the callsite within the same compile unit.
func:
.fill 100
bl func
# CHECK-FIXUP: fixup A - offset: 0, value: func, kind: fixup_loongarch_b26
# CHECK-INSTR: bl -100

.fill 10000
bl func
# CHECK-FIXUP: fixup A - offset: 0, value: func, kind: fixup_loongarch_b26
# CHECK-INSTR: bl -10104

.fill 20888
bl func
# CHECK-FIXUP: fixup A - offset: 0, value: func, kind: fixup_loongarch_b26
# CHECK-INSTR: bl -30996
