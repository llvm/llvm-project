## Test invalid instructions on both loongarch32 and loongarch64 target.

# RUN: not llvm-mc --triple=loongarch32 --mattr=-f %s 2>&1 \
# RUN:         | FileCheck %s --check-prefixes=CHECK,CHECK64
# RUN: not llvm-mc --triple=loongarch64 --mattr=-f %s 2>&1 --defsym=LA64=1 \
# RUN:         | FileCheck %s

## Out of range immediates
## uimm2
bytepick.w $a0, $a0, $a0, -1
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 3]
bytepick.w $a0, $a0, $a0, 4
# CHECK: :[[#@LINE-1]]:27: error: immediate must be an integer in the range [0, 3]

## uimm2_plus1
alsl.w $a0, $a0, $a0, 0
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [1, 4]
alsl.w $a0, $a0, $a0, 5
# CHECK: :[[#@LINE-1]]:23: error: immediate must be an integer in the range [1, 4]

## uimm5
slli.w $a0, $a0, -1
# CHECK: :[[#@LINE-1]]:18: error: immediate must be an integer in the range [0, 31]
srli.w $a0, $a0, -1
# CHECK: :[[#@LINE-1]]:18: error: immediate must be an integer in the range [0, 31]
srai.w $a0, $a0, 32
# CHECK: :[[#@LINE-1]]:18: error: immediate must be an integer in the range [0, 31]
rotri.w $a0, $a0, 32
# CHECK: :[[#@LINE-1]]:19: error: immediate must be an integer in the range [0, 31]
bstrins.w $a0, $a0, 31, -1
# CHECK: :[[#@LINE-1]]:25: error: immediate must be an integer in the range [0, 31]
bstrpick.w $a0, $a0, 32, 0
# CHECK: :[[#@LINE-1]]:22: error: immediate must be an integer in the range [0, 31]
preld -1, $a0, 0
# CHECK: :[[#@LINE-1]]:7: error: immediate must be an integer in the range [0, 31]
preld 32, $a0, 0
# CHECK: :[[#@LINE-1]]:7: error: immediate must be an integer in the range [0, 31]

## uimm12
andi $a0, $a0, -1
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [0, 4095]
xori $a0, $a0, 4096
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [0, 4095]

## uimm12_ori
ori $a0, $a0, 4096
# CHECK: :[[#@LINE-1]]:15: error: operand must be a symbol with modifier (e.g. %abs_lo12) or an integer in the range [0, 4095]

## simm12
slti $a0, $a0, -2049
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-2048, 2047]
sltui $a0, $a0, 2048
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [-2048, 2047]
preld 0, $a0, 2048
# CHECK: :[[#@LINE-1]]:15: error: immediate must be an integer in the range [-2048, 2047]

## simm12_addlike
addi.w $a0, $a0, -2049
# CHECK: :[[#@LINE-1]]:18: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]
ld.b $a0, $a0, 2048
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]
ld.h $a0, $a0, 2048
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]
ld.w $a0, $a0, 2048
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]
ld.bu $a0, $a0, -2049
# CHECK: :[[#@LINE-1]]:17: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]
ld.hu $a0, $a0, -2049
# CHECK: :[[#@LINE-1]]:17: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]
st.b $a0, $a0, 2048
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]
st.h $a0, $a0, 2048
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]
st.w $a0, $a0, -2049
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %pc_lo12) or an integer in the range [-2048, 2047]

## simm14_lsl2
ll.w $a0, $a0, -32772
# CHECK: :[[#@LINE-1]]:16: error: immediate must be a multiple of 4 in the range [-32768, 32764]
ll.w $a0, $a0, -32769
# CHECK: :[[#@LINE-1]]:16: error: immediate must be a multiple of 4 in the range [-32768, 32764]
sc.w $a0, $a0, 32767
# CHECK: :[[#@LINE-1]]:16: error: immediate must be a multiple of 4 in the range [-32768, 32764]
sc.w $a0, $a0, 32768
# CHECK: :[[#@LINE-1]]:16: error: immediate must be a multiple of 4 in the range [-32768, 32764]

## simm16_lsl2
beq $a0, $a0, -0x20004
# CHECK: :[[#@LINE-1]]:15: error: operand must be a symbol with modifier (e.g. %b16) or an integer in the range [-131072, 131068]
bne $a0, $a0, -0x20004
# CHECK: :[[#@LINE-1]]:15: error: operand must be a symbol with modifier (e.g. %b16) or an integer in the range [-131072, 131068]
blt $a0, $a0, -0x1FFFF
# CHECK: :[[#@LINE-1]]:15: error: operand must be a symbol with modifier (e.g. %b16) or an integer in the range [-131072, 131068]
bge $a0, $a0, -0x1FFFF
# CHECK: :[[#@LINE-1]]:15: error: operand must be a symbol with modifier (e.g. %b16) or an integer in the range [-131072, 131068]
bltu $a0, $a0, 0x1FFFF
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %b16) or an integer in the range [-131072, 131068]
bgeu $a0, $a0, 0x1FFFF
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %b16) or an integer in the range [-131072, 131068]
jirl $a0, $a0, 0x20000
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %b16) or an integer in the range [-131072, 131068]

## simm20
pcaddi $a0, -0x80001
# CHECK: :[[#@LINE-1]]:13: error: immediate must be an integer in the range [-524288, 524287]
pcaddu12i $a0, 0x80000
# CHECK: :[[#@LINE-1]]:16: error: immediate must be an integer in the range [-524288, 524287]

## simm20_lu12iw
lu12i.w $a0, -0x80001
# CHECK: :[[#@LINE-1]]:14: error: operand must be a symbol with modifier (e.g. %abs_hi20) or an integer in the range [-524288, 524287]

## simm20_pcalau12i
pcalau12i $a0, 0x80000
# CHECK: :[[#@LINE-1]]:16: error: operand must be a symbol with modifier (e.g. %pc_hi20) or an integer in the range [-524288, 524287]

## simm21_lsl2
beqz $a0, -0x400001
# CHECK: :[[#@LINE-1]]:11: error: operand must be a symbol with modifier (e.g. %b21) or an integer in the range [-4194304, 4194300]
bnez $a0, -0x3FFFFF
# CHECK: :[[#@LINE-1]]:11: error: operand must be a symbol with modifier (e.g. %b21) or an integer in the range [-4194304, 4194300]
beqz $a0, 0x3FFFFF
# CHECK: :[[#@LINE-1]]:11: error: operand must be a symbol with modifier (e.g. %b21) or an integer in the range [-4194304, 4194300]
bnez $a0, 0x400000
# CHECK: :[[#@LINE-1]]:11: error: operand must be a symbol with modifier (e.g. %b21) or an integer in the range [-4194304, 4194300]

## simm26_lsl2
b -0x8000001
# CHECK: :[[#@LINE-1]]:3: error: operand must be a bare symbol name or an immediate must be a multiple of 4 in the range [-134217728, 134217724]
b 0x1
# CHECK: :[[#@LINE-1]]:3: error: operand must be a bare symbol name or an immediate must be a multiple of 4 in the range [-134217728, 134217724]
bl 0x7FFFFFF
# CHECK: :[[#@LINE-1]]:4: error: operand must be a bare symbol name or an immediate must be a multiple of 4 in the range [-134217728, 134217724]
bl 0x8000000
# CHECK: :[[#@LINE-1]]:4: error: operand must be a bare symbol name or an immediate must be a multiple of 4 in the range [-134217728, 134217724]

## Invalid mnemonics
nori $a0, $a0, 0
# CHECK: :[[#@LINE-1]]:1: error: unrecognized instruction mnemonic
andni $a0, $a0, 0
# CHECK: :[[#@LINE-1]]:1: error: unrecognized instruction mnemonic
orni $a0, $a0, 0
# CHECK: :[[#@LINE-1]]:1: error: unrecognized instruction mnemonic

## Invalid register names
add.w $foo, $a0, $a0
# CHECK: :[[#@LINE-1]]:8: error: invalid operand for instruction
sub.w $a8, $a0, $a0
# CHECK: :[[#@LINE-1]]:8: error: invalid operand for instruction
addi.w $x0, $a0, 0
# CHECK: :[[#@LINE-1]]:9: error: invalid operand for instruction
alsl.w $t9, $a0, $a0, 1
# CHECK: :[[#@LINE-1]]:9: error: invalid operand for instruction
lu12i.w $s10, 0
# CHECK: :[[#@LINE-1]]:10: error: invalid operand for instruction

.ifndef LA64
## LoongArch64 mnemonics
add.d $a0, $a0, $a0
# CHECK64: :[[#@LINE-1]]:1: error: instruction requires the following: LA64 Basic Integer and Privilege Instruction Set
addi.d $a0, $a0, 0
# CHECK64: :[[#@LINE-1]]:1: error: instruction requires the following: LA64 Basic Integer and Privilege Instruction Set
.endif

## Invalid operand types
slt $a0, $a0, 0
# CHECK: :[[#@LINE-1]]:15: error: invalid operand for instruction
slti $a0, 0, 0
# CHECK: :[[#@LINE-1]]:11: error: invalid operand for instruction

## Too many operands
andi $a0, $a0, 0, 0
# CHECK: :[[#@LINE-1]]:19: error: invalid operand for instruction

## Too few operands
and $a0, $a0
# CHECK: :[[#@LINE-1]]:1: error: too few operands for instruction
andi $a0, $a0
# CHECK: :[[#@LINE-1]]:1: error: too few operands for instruction

## Instructions outside the base integer ISA
## TODO: Test instructions in LSX/LASX/LBT/LVZ after their introduction.

## Floating-Point mnemonics
fadd.s $fa0, $fa0, $fa0
# CHECK:   :[[#@LINE-1]]:1: error: instruction requires the following: 'F' (Single-Precision Floating-Point)
fadd.d $fa0, $fa0, $fa0
# CHECK:   :[[#@LINE-1]]:1: error: instruction requires the following: 'D' (Double-Precision Floating-Point)

## Using floating point registers when integer registers are expected
sll.w $a0, $a0, $fa0
# CHECK: :[[#@LINE-1]]:18: error: invalid operand for instruction

## msbw < lsbw
# CHECK: :[[#@LINE+1]]:21: error: msb is less than lsb
bstrins.w $a0, $a0, 1, 2
# CHECK:            ^~~~

# CHECK: :[[#@LINE+1]]:22: error: msb is less than lsb
bstrpick.w $a0, $a0, 30, 31
# CHECK:             ^~~~~~
