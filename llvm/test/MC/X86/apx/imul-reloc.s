// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  subw $foo, %ax, %ax                 // CHECK-NEXT:     R_X86_64_16
{nf}    subw $foo, %ax, %ax                 // CHECK-NEXT:     R_X86_64_16
{evex}  subw $foo, 123(%r8,%rax,4), %ax     // CHECK-NEXT:     R_X86_64_16
{nf}    subw $foo, 123(%r8,%rax,4), %ax     // CHECK-NEXT:     R_X86_64_16
{evex}  subl $foo, %eax, %eax               // CHECK-NEXT:     R_X86_64_32
{nf}    subl $foo, %eax, %eax               // CHECK-NEXT:     R_X86_64_32
{evex}  subl $foo, 123(%r8,%rax,4), %eax    // CHECK-NEXT:     R_X86_64_32
{nf}    subl $foo, 123(%r8,%rax,4), %eax    // CHECK-NEXT:     R_X86_64_32
{evex}  subq $foo, %rax, %rax               // CHECK-NEXT:     R_X86_64_32S
{nf}    subq $foo, %rax, %rax               // CHECK-NEXT:     R_X86_64_32S
{evex}  subq $foo, 123(%r8,%rax,4), %rax    // CHECK-NEXT:     R_X86_64_32S
{nf}    subq $foo, 123(%r8,%rax,4), %rax    // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
