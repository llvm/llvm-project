// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
ccmpbb	{dfv=of}	$foo, %bl               // CHECK-NEXT:     R_X86_64_8
ccmpbb	{dfv=of}	$foo, 123(%r8,%rax,4)   // CHECK-NEXT:     R_X86_64_8
ccmpbw	{dfv=of}	$foo, %bx               // CHECK-NEXT:     R_X86_64_16
ccmpbw	{dfv=of}	$foo, 123(%r8,%rax,4)   // CHECK-NEXT:     R_X86_64_16
ccmpbl	{dfv=of}	$foo, %ebx              // CHECK-NEXT:     R_X86_64_32
ccmpbl	{dfv=of}	$foo, 123(%r8,%rax,4)   // CHECK-NEXT:     R_X86_64_32
ccmpbq	{dfv=of}	$foo, %rbx              // CHECK-NEXT:     R_X86_64_32S
ccmpbq	{dfv=of}	$foo, 123(%r8,%rax,4)   // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
