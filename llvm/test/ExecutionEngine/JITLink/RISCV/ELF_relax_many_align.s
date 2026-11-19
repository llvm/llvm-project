## Instrumented clang has many alignment edges mixed with call edges in one
## code block. Removing alignment edges must preserve all remaining fixups.
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.o
# RUN: llvm-jitlink -noexec -slab-allocate 1Mb -slab-address 0x0 \
# RUN:     -slab-page-size 4096 -check %s %t.o

  .text
  .globl main, last, target
main:
  .rept 4096
  call target
  .balign 16
  .endr
last:
  call target
  .balign 16
target:
  ret
  .size main, .-main
  .size last, target-last
  .size target, 4

# jitlink-check: last - main = 65536
# jitlink-check: target - last = 16
# jitlink-check: decode_operand(main, 1) = (target - main)
# jitlink-check: decode_operand(last, 1) = (target - last)
# jitlink-check: *{4}(last + 4) = 0x13
# jitlink-check: *{4}(last + 8) = 0x13
# jitlink-check: *{4}(last + 12) = 0x13
