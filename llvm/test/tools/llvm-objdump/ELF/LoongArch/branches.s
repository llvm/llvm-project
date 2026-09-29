# RUN: llvm-mc --triple=loongarch32 --filetype=obj < %s | \
# RUN:   llvm-objdump -d --no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc --triple=loongarch64 --filetype=obj < %s | \
# RUN:   llvm-objdump -d --no-show-raw-insn - | FileCheck %s

# CHECK-LABEL: <foo>:
foo:
# CHECK: beq $a0, $a1, 108 <foo+0x6c>
beq $a0, $a1, .Llocal
# CHECK: bne $a0, $a1, 104 <foo+0x6c>
bne $a0, $a1, .Llocal
# CHECK: blt $a0, $a1, 100 <foo+0x6c>
blt $a0, $a1, .Llocal
# CHECK: bltu $a0, $a1, 96 <foo+0x6c>
bltu $a0, $a1, .Llocal
# CHECK: bge $a0, $a1, 92 <foo+0x6c>
bge $a0, $a1, .Llocal
# CHECK: bgeu $a0, $a1, 88 <foo+0x6c>
bgeu $a0, $a1, .Llocal
# CHECK: beqz $a0, 84 <foo+0x6c>
beqz $a0, .Llocal
# CHECK: bnez $a0, 80 <foo+0x6c>
bnez $a0, .Llocal
# CHECK: bceqz $fcc6, 76 <foo+0x6c>
bceqz $fcc6, .Llocal
# CHECK: bcnez $fcc6, 72 <foo+0x6c>
bcnez $fcc6, .Llocal

# CHECK: beq $a0, $a1, 76 <bar>
beq $a0, $a1, bar
# CHECK: bne $a0, $a1, 72 <bar>
bne $a0, $a1, bar
# CHECK: blt $a0, $a1, 68 <bar>
blt $a0, $a1, bar
# CHECK: bltu $a0, $a1, 64 <bar>
bltu $a0, $a1, bar
# CHECK: bge $a0, $a1, 60 <bar>
bge $a0, $a1, bar
# CHECK: bgeu $a0, $a1, 56 <bar>
bgeu $a0, $a1, bar
# CHECK: beqz $a0, 52 <bar>
beqz $a0, bar
# CHECK: bnez $a0, 48 <bar>
bnez $a0, bar
# CHECK: bceqz $fcc6, 44 <bar>
bceqz $fcc6, bar
# CHECK: bcnez $fcc6, 40 <bar>
bcnez $fcc6, bar

# CHECK: b 28 <foo+0x6c>
b .Llocal
# CHECK: b 32 <bar>
b bar

# CHECK: bl 20 <foo+0x6c>
bl .Llocal
# CHECK: bl 24 <bar>
bl bar

# CHECK: jirl $zero, $a0, 4{{$}}
jirl $zero, $a0, 4
# CHECK: jirl $ra, $a0, 4{{$}}
jirl $ra, $a0, 4
# CHECK: ret
ret

.Llocal:
# CHECK: 6c: nop
# CHECK: nop
nop
nop

# CHECK-LABEL: <bar>:
bar:
# CHECK: 74: nop
nop
