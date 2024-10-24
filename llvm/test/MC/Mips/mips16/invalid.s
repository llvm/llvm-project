# REQUIRES: mips-registered-target
# RUN: not llvm-mc -arch=mipsel -mcpu=mips32r2 -mattr=+mips16 %s 2> %t
# RUN: FileCheck %s < %t

# Use NOP to check instructions that do not take arguments.

$label:
  nop 4         # CHECK: :[[#@LINE]]:[[#]]: error: invalid operand for instruction
  nop $4        # CHECK: :[[#@LINE]]:[[#]]: error: invalid operand for instruction
  nop $label    # CHECK: :[[#@LINE]]:[[#]]: error: invalid operand for instruction

# These are MIPS instructions, but are not usable in MIPS16 mode.

and       $s1,$v0,$12        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
andi      $2, $3, 4          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
cache     1, 16($5)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
clz       $sp,$s0            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
di                           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
ehb                          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
ei                           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
eret                         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lwl       $s1,120($v1)       # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lwr       $16,68($17)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
mfc0      $3,$15,1           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
mtc0      $4,$15,1           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
pref      1, 8($5)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
ssnop                        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sync                         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sync      1                  # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
syscall                      # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
syscall   256                # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
teq $zero, $3                # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
teq $5, $7, 620              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
tlbp                         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
tlbr                         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
tlbwi                        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
tlbwr                        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled

# These instructions are valid in MIPS16, but use registers not 
# avaialble in MIPS16 mode.

addiu $23, $17, -759         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
addiu $17, $23, -759         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
addu $7, $0, $2              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
addu $7, $16, $zero          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
addu $s5, $16, $2            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
div $7, $s6                  # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
div $t0, $s0                 # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
divu $2, $fp                 # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
divu $t8, $16                # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
jr $k0                       # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lb $v1, 22774($t5)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lb $k1, 22774($a0)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lbu $3, 17($22)              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lbu $20, 17($6)              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lh $7, 11699($28)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lh $29, 11699($4)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lhu $5, 20($8)               # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lhu $t1, 20($v0)             # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
li $15, 3257                 # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $s1, 60($zero)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $s2, 60($a0)              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $sp, 23037($a0)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
mfhi $13                     # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
mflo $10                     # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sb $s0, 1610($11)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sb $12, 1610($v0)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sh $5, 20990($at)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sh $t2, 20990($2)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sll $k1, $a3, 5              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sll $v1, $k0, 5              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sra $18, $6, 11              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sra $s0, $19, 11             # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
srl $4, $21, 2               # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
srl $27, $2, 2               # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
subu $17, $4, $0             # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
subu $17, $sp, $a1           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
subu $fp, $4, $a1            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sw $sp, 104($7)              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sw $4, 104($t6)              # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled

# These instructions are valid in MIPS16 and use MIPS16 registers, 
# but the immediates are accepted in MIPS32 mode only.
# FIXME: Add large immediate support to MIPS16.
#       See MipsAsmParser::ExpandMem16Instruction() and expandAliasImmediate().
addiu $v0, $s0, -16385       # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
addiu $a2, $a1, 16384        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
addiu $v0, 32768             # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
addiu $sp, 32768             # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
addiu $5, $sp, 32768         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lb $a0, -32769($2)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lb $v1, 32768($a0)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lbu $4, -32769($16)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lbu $4, 32768($17)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lh $a2, -32769($a1)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lh $7, 32768($4)             # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lhu $s1, -32769($v0)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lhu $s0, 32769($2)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
li $16, 65536                # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $a0, -32769($v0)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $4, 32768($7)             # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $6, -32769($pc)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $2, 32768($pc)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $17, -32769($sp)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
lw $s1, 32768($sp)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sb $a0, -32769($7)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sb $s0, 32768($v0)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sh $16, -32769($s1)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sh $5, 32768($2)             # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sw $s0, -32769($s0)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sw $a3, 32768($7)            # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sw $17, -32769($sp)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sw $s1, 32768($sp)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sw $ra, -32769($sp)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
sw $ra, 32768($sp)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
li $a3, -1                   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled

# These instructions are valid in MIPS16 and use MIPS16 registers,
# but have out of range immediate values.

addiu $a3, -32769            # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
addiu $a2, $pc, -32769       # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
addiu $4, $pc, 32768         # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
addiu $sp, -32769            # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
addiu $16, $sp, -32769       # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
addiu $v0, $s0, -65537       # CHECK: :[[#@LINE]]:[[#]]: error: expected 15-bit signed immediate
addiu $a2, $a1, 65536        # CHECK: :[[#@LINE]]:[[#]]: error: expected 15-bit signed immediate
addiu $v0, 65536             # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
addiu $sp, 65536             # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
addiu $5, $sp, 65536         # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
asmacro 8, 15, 6, 7, 21, 3   # CHECK: :[[#@LINE]]:[[#]]: error: expected 3-bit unsigned immediate
asmacro 4, 32, 6, 7, 21, 3   # CHECK: :[[#@LINE]]:[[#]]: error: expected 5-bit unsigned immediate
asmacro 4, 15, 8, 7, 21, 3   # CHECK: :[[#@LINE]]:[[#]]: error: expected 3-bit unsigned immediate
asmacro 4, 15, 6, 8, 21, 3   # CHECK: :[[#@LINE]]:[[#]]: error: expected 3-bit unsigned immediate
asmacro 4, 15, 6, 7, 32, 3   # CHECK: :[[#@LINE]]:[[#]]: error: expected 5-bit unsigned immediate
asmacro 4, 15, 6, 7, 21, 8   # CHECK: :[[#@LINE]]:[[#]]: error: expected 3-bit unsigned immediate
b -131074                    # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
b 131072                     # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
beqz $a0, -131074            # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
beqz $a0, 131072             # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
bnez $v1, -131074            # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
bnez $v1, 131072             # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
bteqz -131074                # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
bteqz 131072                 # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
btnez -131074                # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
btnez 131072                 # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
cmpi $a0, -1                 # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit unsigned immediate
cmpi $5, 65536               # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit unsigned immediate
jal -256                     # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
jal 268435456                # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
jalx -256                    # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
jalx 268435456               # CHECK: :[[#@LINE]]:[[#]]: error: branch target out of range
sll $s0, $2, 32              # CHECK: :[[#@LINE]]:[[#]]: error: expected 5-bit unsigned immediate
slti $a1, -32769             # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
slti $16, 32768              # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
sltiu $5, -32769             # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
sltiu $5, 65536              # CHECK: :[[#@LINE]]:[[#]]: error: expected 16-bit signed immediate
sra $s0, $6, 32              # CHECK: :[[#@LINE]]:[[#]]: error: expected 5-bit unsigned immediate
srl $v0, $17, 32             # CHECK: :[[#@LINE]]:[[#]]: error: expected 5-bit unsigned immediate

# Unalingned branches and jumps

b -851                       # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
b 71                         # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
b -24193                     # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
b 30343                      # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
beqz $17, -211               # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
beqz $v1, 207                # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
beqz $s1, -39467             # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
beqz $a0, 26545              # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
bnez $6, -169                # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
bnez $a3, 139                # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
bnez $3, -10353              # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
bnez $s1, 61261              # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
bteqz -169                   # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
bteqz 47                     # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
bteqz -18163                 # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
bteqz 61095                  # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
btnez -161                   # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
btnez 71                     # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
btnez -29969                 # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
btnez 44113                  # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
jal 254                      # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address
jalx 254                     # CHECK: :[[#@LINE]]:[[#]]: error: branch to misaligned address

# Try malformed versions of SAVE and RESTORE.

restore -8                     # CHECK: :[[#@LINE]]:[[#]]: error: frame size must be in range 0 .. 2040 and a multiple of 8
restore 2048                   # CHECK: :[[#@LINE]]:[[#]]: error: frame size must be in range 0 .. 2040 and a multiple of 8
restore $24, 72                # CHECK: :[[#@LINE]]:[[#]]: error: only registers $4-7, $16-23, $30, and $31 can be used
restore 72, $27                # CHECK: :[[#@LINE]]:[[#]]: error: only registers $4-7 can be saved as static registers
restore 80, $28                # CHECK: :[[#@LINE]]:[[#]]: error: only registers $4-7 can be saved as static registers
restore $17, 160, $17          # CHECK: :[[#@LINE]]:[[#]]: error: only registers $4-7 can be saved as static registers
restore $16, $17, 720, 800     # CHECK: :[[#@LINE]]:[[#]]: error: expected static register
restore 800, $4, $5, 720       # CHECK: :[[#@LINE]]:[[#]]: error: expected static register
restore $4, $5, 120, $4, $5    # CHECK: :[[#@LINE]]:[[#]]: error: registers cannot be both in saved and static lists
restore $18-$21-, 880          # CHECK: :[[#@LINE]]:[[#]]: error: unexpected token, expected comma
restore $18-$21-$23, 960       # CHECK: :[[#@LINE]]:[[#]]: error: unexpected token, expected comma
restore $18-400                # CHECK: :[[#@LINE]]:[[#]]: error: expected end of register range
restore 240($18)               # CHECK: :[[#@LINE]]:[[#]]: error: unexpected token, expected comma
save -8                        # CHECK: :[[#@LINE]]:[[#]]: error: frame size must be in range 0 .. 2040 and a multiple of 8
save 2048                      # CHECK: :[[#@LINE]]:[[#]]: error: frame size must be in range 0 .. 2040 and a multiple of 8
save $24, 72                   # CHECK: :[[#@LINE]]:[[#]]: error: only registers $4-7, $16-23, $30, and $31 can be used
save 72, $27                   # CHECK: :[[#@LINE]]:[[#]]: error: only registers $4-7 can be saved as static registers
save 80, $28                   # CHECK: :[[#@LINE]]:[[#]]: error: only registers $4-7 can be saved as static registers
save $17, 160, $17             # CHECK: :[[#@LINE]]:[[#]]: error: only registers $4-7 can be saved as static registers
save $16, $17, 720, 800        # CHECK: :[[#@LINE]]:[[#]]: error: expected static register
save 800, $4, $5, 720          # CHECK: :[[#@LINE]]:[[#]]: error: expected static register
save $4, $5, 120, $4, $5       # CHECK: :[[#@LINE]]:[[#]]: error: registers cannot be both in saved and static lists
save $18-$21-, 880             # CHECK: :[[#@LINE]]:[[#]]: error: unexpected token, expected comma
save $18-$21-$23, 960          # CHECK: :[[#@LINE]]:[[#]]: error: unexpected token, expected comma
save $18-400                   # CHECK: :[[#@LINE]]:[[#]]: error: expected end of register range
save 240($18)                  # CHECK: :[[#@LINE]]:[[#]]: error: unexpected token, expected comma
