// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  adcb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
adcb $foo, %al, %bl                         // CHECK-NEXT:     R_X86_64_8
{evex}  adcb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8   
adcb $foo, 123(%r8,%rax,4), %bl             // CHECK-NEXT:     R_X86_64_8
{evex}  adcw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
adcw $foo, %ax, %bx                         // CHECK-NEXT:     R_X86_64_16
{evex}  adcw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
adcw $foo, 123(%r8,%rax,4), %bx             // CHECK-NEXT:     R_X86_64_16
{evex}  adcl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
adcl $foo, %eax, %ebx                       // CHECK-NEXT:     R_X86_64_32
{evex}  adcl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
adcl $foo, 123(%r8,%rax,4), %ebx            // CHECK-NEXT:     R_X86_64_32
{evex}  adcq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
adcq $foo, %rax, %rbx                       // CHECK-NEXT:     R_X86_64_32S
{evex}  adcq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
adcq $foo, 123(%r8,%rax,4), %rbx            // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
