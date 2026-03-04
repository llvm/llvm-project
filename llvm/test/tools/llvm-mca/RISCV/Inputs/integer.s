# Input instructions for the 'I' and 'Zicsr' extensions.

# Integer Register-Immediate Instructions
addi a0, a0, 1
addiw a0, a0, 1
slti a0, a0, 1
sltiu a0, a0, 1

andi a0, a0, 1
ori a0, a0, 1
xori a0, a0, 1

slli a0, a0, 1
srli a0, a0, 1
srai a0, a0, 1
slliw a0, a0, 1
srliw a0, a0, 1
sraiw a0, a0, 1

lui a0, 1
auipc a1, 1

# Integer Register-Register Operations
add a0, a0, a1
addw a0, a0, a0
slt a0, a0, a0
sltu a0, a0, a0

and a0, a0, a0
or a0, a0, a0
xor a0, a0, a0

sll a0, a0, a0
srl a0, a0, a0
sra a0, a0, a0
sllw a0, a0, a0
srlw a0, a0, a0
sraw a0, a0, a0

sub a0, a0, a0
subw a0, a0, a0

# Control Transfer Instructions

## Unconditional Jumps
jal a0, 1f
1:
jalr a0
beq a0, a0, 1f
1:
bne a0, a0, 1f
1:
blt a0, a0, 1f
1:
bltu a0, a0, 1f
1:
bge a0, a0, 1f
1:
bgeu a0, a0, 1f
1:
add a0, a0, a0

# Load and Store Instructions
lb t0, 0(a0)
lbu t0, 0(a0)
lh t0, 0(a0)
lhu t0, 0(a0)
lw t0, 0(a0)
lwu t0, 0(a0)
ld t0, 0(a0)

sb t0, 0(a0)
sh t0, 0(a0)
sw t0, 0(a0)
sd t0, 0(a0)

# Zicsr
csrrw t0, 0xfff, t1
csrrs s3, 0x001, s5
csrrc sp, 0x000, ra
csrrwi a5, 0x000, 0
csrrsi t2, 0xfff, 31
csrrci t1, 0x140, 5
