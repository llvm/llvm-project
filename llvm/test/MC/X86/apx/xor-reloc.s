// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  xorb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
{nf}    xorb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
xorb $foo, %al, %bl                         // CHECK-NEXT:     R_X86_64_8
{nf}    xorb $foo, %al, %bl                 // CHECK-NEXT:     R_X86_64_8
{evex}  xorb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8   
{nf}    xorb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8
xorb $foo, 123(%r8,%rax,4), %bl             // CHECK-NEXT:     R_X86_64_8
{nf}    xorb $foo, 123(%r8,%rax,4), %bl     // CHECK-NEXT:     R_X86_64_8
{evex}  xorw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
{nf}    xorw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
xorw $foo, %ax, %bx                         // CHECK-NEXT:     R_X86_64_16
{nf}    xorw $foo, %ax, %bx                 // CHECK-NEXT:     R_X86_64_16
{evex}  xorw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
{nf}    xorw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
xorw $foo, 123(%r8,%rax,4), %bx             // CHECK-NEXT:     R_X86_64_16
{nf}    xorw $foo, 123(%r8,%rax,4), %bx     // CHECK-NEXT:     R_X86_64_16
{evex}  xorl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
{nf}    xorl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
xorl $foo, %eax, %ebx                       // CHECK-NEXT:     R_X86_64_32
{nf}    xorl $foo, %eax, %ebx               // CHECK-NEXT:     R_X86_64_32
{evex}  xorl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
{nf}    xorl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
xorl $foo, 123(%r8,%rax,4), %ebx            // CHECK-NEXT:     R_X86_64_32
{nf}    xorl $foo, 123(%r8,%rax,4), %ebx    // CHECK-NEXT:     R_X86_64_32
{evex}  xorq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
{nf}    xorq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
xorq $foo, %rax, %rbx                       // CHECK-NEXT:     R_X86_64_32S
{nf}    xorq $foo, %rax, %rbx               // CHECK-NEXT:     R_X86_64_32S
{evex}  xorq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
{nf}    xorq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
xorq $foo, 123(%r8,%rax,4), %rbx            // CHECK-NEXT:     R_X86_64_32S
{nf}    xorq $foo, 123(%r8,%rax,4), %rbx    // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
