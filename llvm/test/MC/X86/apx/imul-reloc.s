// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  imulw $foo, %ax, %ax                 // CHECK-NEXT:     R_X86_64_16
{nf}    imulw $foo, %ax, %ax                 // CHECK-NEXT:     R_X86_64_16
{evex}  imulw $foo, 123(%r8,%rax,4), %ax     // CHECK-NEXT:     R_X86_64_16
{nf}    imulw $foo, 123(%r8,%rax,4), %ax     // CHECK-NEXT:     R_X86_64_16
{evex}  imull $foo, %eax, %eax               // CHECK-NEXT:     R_X86_64_32
{nf}    imull $foo, %eax, %eax               // CHECK-NEXT:     R_X86_64_32
{evex}  imull $foo, 123(%r8,%rax,4), %eax    // CHECK-NEXT:     R_X86_64_32
{nf}    imull $foo, 123(%r8,%rax,4), %eax    // CHECK-NEXT:     R_X86_64_32
{evex}  imulq $foo, %rax, %rax               // CHECK-NEXT:     R_X86_64_32S
{nf}    imulq $foo, %rax, %rax               // CHECK-NEXT:     R_X86_64_32S
{evex}  imulq $foo, 123(%r8,%rax,4), %rax    // CHECK-NEXT:     R_X86_64_32S
{nf}    imulq $foo, 123(%r8,%rax,4), %rax    // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
