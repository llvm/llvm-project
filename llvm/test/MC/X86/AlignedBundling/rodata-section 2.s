# RUN: llvm-mc -triple=i686-nacl -filetype=obj %s -o - \
# RUN:    | llvm-objdump --no-print-imm-hex -d --no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -triple=i686-nacl -filetype=obj -mc-relax-all %s -o - \
# RUN:    | llvm-objdump --no-print-imm-hex -d --no-show-raw-insn - | FileCheck %s

  .bundle_align_mode 5
  .text
  .align	32, 0x90
# CHECK: 0: movl $14, 8(%esp)
  movl	$.str2, 8(%esp)
# CHECK: 8: movl $7, 4(%esp)
  movl	$.str1, 4(%esp)
# CHECK: 10: movl $0, (%esp)
  movl	$.str, (%esp)

  .type	.str,@object
  .section	.rodata,"a",@progbits
.str:
  .asciz	"hello1"
  .size	.str, 7

  .type	.str1,@object
.str1:
  .asciz	"hello2"
  .size	.str1, 7

  .type	.str2,@object
.str2:
  .asciz	"hello3"
  .size	.str2, 7
