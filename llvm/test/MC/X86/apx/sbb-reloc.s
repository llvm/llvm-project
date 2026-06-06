// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  sbbb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
sbbb $foo, %al, %bl                         // CHECK-NEXT:     R_X86_64_8
{evex}  sbbb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8   
sbbb $foo, 123(%r8,%rax,4), %bl             // CHECK-NEXT:     R_X86_64_8
{evex}  sbbw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
sbbw $foo, %ax, %bx                         // CHECK-NEXT:     R_X86_64_16
{evex}  sbbw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
sbbw $foo, 123(%r8,%rax,4), %bx             // CHECK-NEXT:     R_X86_64_16
{evex}  sbbl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
sbbl $foo, %eax, %ebx                       // CHECK-NEXT:     R_X86_64_32
{evex}  sbbl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
sbbl $foo, 123(%r8,%rax,4), %ebx            // CHECK-NEXT:     R_X86_64_32
{evex}  sbbq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
sbbq $foo, %rax, %rbx                       // CHECK-NEXT:     R_X86_64_32S
{evex}  sbbq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
sbbq $foo, 123(%r8,%rax,4), %rbx            // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
