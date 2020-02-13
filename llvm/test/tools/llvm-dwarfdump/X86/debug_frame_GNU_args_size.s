# RUN: llvm-mc %s -filetype=obj -triple=i686-pc-linux -o %t
# RUN: llvm-dwarfdump -v %t | FileCheck %s

# CHECK:      .eh_frame contents:
# CHECK:        00000018 00000010 0000001c FDE cie=00000000 pc=00000000...00000000
# CHECK-NEXT:     DW_CFA_GNU_args_size: +16
# CHECK-NEXT:     DW_CFA_nop:

.text
.globl foo
.type  foo,@function
foo:
 .cfi_startproc
 .cfi_escape 0x2e, 0x10
 .cfi_endproc
