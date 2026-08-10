// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  andb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
{nf}    andb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
andb $foo, %al, %bl                         // CHECK-NEXT:     R_X86_64_8
{nf}    andb $foo, %al, %bl                 // CHECK-NEXT:     R_X86_64_8
{evex}  andb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8   
{nf}    andb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8
andb $foo, 123(%r8,%rax,4), %bl             // CHECK-NEXT:     R_X86_64_8
{nf}    andb $foo, 123(%r8,%rax,4), %bl     // CHECK-NEXT:     R_X86_64_8
{evex}  andw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
{nf}    andw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
andw $foo, %ax, %bx                         // CHECK-NEXT:     R_X86_64_16
{nf}    andw $foo, %ax, %bx                 // CHECK-NEXT:     R_X86_64_16
{evex}  andw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
{nf}    andw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
andw $foo, 123(%r8,%rax,4), %bx             // CHECK-NEXT:     R_X86_64_16
{nf}    andw $foo, 123(%r8,%rax,4), %bx     // CHECK-NEXT:     R_X86_64_16
{evex}  andl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
{nf}    andl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
andl $foo, %eax, %ebx                       // CHECK-NEXT:     R_X86_64_32
{nf}    andl $foo, %eax, %ebx               // CHECK-NEXT:     R_X86_64_32
{evex}  andl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
{nf}    andl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
andl $foo, 123(%r8,%rax,4), %ebx            // CHECK-NEXT:     R_X86_64_32
{nf}    andl $foo, 123(%r8,%rax,4), %ebx    // CHECK-NEXT:     R_X86_64_32
{evex}  andq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
{nf}    andq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
andq $foo, %rax, %rbx                       // CHECK-NEXT:     R_X86_64_32S
{nf}    andq $foo, %rax, %rbx               // CHECK-NEXT:     R_X86_64_32S
{evex}  andq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
{nf}    andq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
andq $foo, 123(%r8,%rax,4), %rbx            // CHECK-NEXT:     R_X86_64_32S
{nf}    andq $foo, 123(%r8,%rax,4), %rbx    // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
