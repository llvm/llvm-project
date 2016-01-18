# RUN: not llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -arch=mips 2>&1 -filetype=obj | FileCheck %s
#
# CHECK: LLVM ERROR: out of range PC16 fixup

.text
  b foo
  .space 131072 - 8, 1  # -8 = size of b instr plus size of automatically inserted nop
  nop                   # This instr makes the branch too long to fit into a 18-bit offset
foo:
  add $0,$0,$0
