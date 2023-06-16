// RUN: llvm-mc -filetype=obj %s -o - -triple x86_64-pc-linux | llvm-objdump -s - | FileCheck %s
// RUN: not llvm-mc -filetype=obj -triple x86_64 --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

// CHECK: Contents of section .text:
// CHECK-NEXT: 0000 03042502 00000003 04250100 0000ebf7
.text 1
add 1, %eax
jmp label
.subsection
add 2, %eax
label:

// CHECK-NOT: Contents of section .rela.text:

// CHECK: Contents of section .data:
// CHECK-NEXT: 0000 01030402 74657374
.data
l0:
.byte 1
.subsection 1+1
l1:
.byte 2
l2:
.subsection l2-l1
.byte l1-l0
.subsection 3
.ascii "test"
.previous
.byte 4

// CHECK: Contents of section test:
// CHECK-NEXT: 0000 010302
.section test
.byte 1
.pushsection test, 1
.byte 2
.popsection
.byte 3


.ifdef ERR
// ERR: :[[#@LINE+1]]:13: error: cannot evaluate subsection number
.subsection l2

// ERR: <unknown>:0: error: subsection number -1 is not within [0,2147483647]
.subsection 0-1
.subsection 2147483647
// ERR: <unknown>:0: error: subsection number 2147483648 is not within [0,2147483647]
.subsection 2147483648
.endif
