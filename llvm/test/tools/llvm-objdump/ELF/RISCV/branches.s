# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s | \
# RUN:     llvm-objdump -d -M no-aliases --no-show-raw-insn - | \
# RUN:     FileCheck %s

# CHECK-LABEL: <foo>:
foo:
# CHECK: beq t0, t1, 0x58 <foo+0x58>
beq t0, t1, .Llocal
# CHECK: bne t0, t1, 0x58 <foo+0x58>
bne t0, t1, .Llocal
# CHECK: blt t0, t1, 0x58 <foo+0x58>
blt t0, t1, .Llocal
# CHECK: bge t0, t1, 0x58 <foo+0x58>
bge t0, t1, .Llocal
# CHECK: bltu t0, t1, 0x58 <foo+0x58>
bltu t0, t1, .Llocal
# CHECK: bgeu t0, t1, 0x58 <foo+0x58>
bgeu t0, t1, .Llocal

# CHECK: c.beqz a0, 0x58 <foo+0x58>
beq a0, zero, .Llocal
# CHECK: c.bnez a0, 0x58 <foo+0x58>
bne a0, zero, .Llocal

# CHECK: beq t0, t1, 0x60 <bar>
beq t0, t1, bar
# CHECK: bne t0, t1, 0x60 <bar>
bne t0, t1, bar
# CHECK: blt t0, t1, 0x60 <bar>
blt t0, t1, bar
# CHECK: bge t0, t1, 0x60 <bar>
bge t0, t1, bar
# CHECK: bltu	t0, t1, 0x60 <bar>
bltu t0, t1, bar
# CHECK: bgeu	t0, t1, 0x60 <bar>
bgeu t0, t1, bar

# CHECK: c.beqz	a0, 0x60 <bar>
beq a0, zero, bar
# CHECK: c.bnez	a0, 0x60 <bar>
bne a0, zero, bar

# CHECK: jal t0, 0x58 <foo+0x58>
jal t0, .Llocal
# CHECK: c.jal 0x58 <foo+0x58>
c.jal .Llocal

# CHECK: c.j 0x58 <foo+0x58>
c.j .Llocal

# CHECK: jal t0, 0x60 <bar>
jal t0, bar
# CHECK: c.jal 0x60 <bar>
c.jal bar

# CHECK: c.j 0x60 <bar>
c.j bar

# CHECK: auipc ra, 0x0
# CHECK: jalr	ra, 0x10(ra) <foo+0x58>
call .Llocal

# CHECK: auipc ra, 0x0
# CHECK: jalr	ra, 0x10(ra) <bar>
call bar

.Llocal:
# CHECK: 58: c.nop
# CHECK: c.nop
# CHECK: c.nop
# CHECK: c.nop
nop
nop
nop
nop

# CHECK-LABEL: <bar>:
bar:
# CHECK: 60: c.nop
nop
