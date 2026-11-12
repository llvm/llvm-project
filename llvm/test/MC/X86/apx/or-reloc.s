// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  orb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
{nf}    orb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
orb $foo, %al, %bl                         // CHECK-NEXT:     R_X86_64_8
{nf}    orb $foo, %al, %bl                 // CHECK-NEXT:     R_X86_64_8
{evex}  orb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8   
{nf}    orb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8
orb $foo, 123(%r8,%rax,4), %bl             // CHECK-NEXT:     R_X86_64_8
{nf}    orb $foo, 123(%r8,%rax,4), %bl     // CHECK-NEXT:     R_X86_64_8
{evex}  orw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
{nf}    orw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
orw $foo, %ax, %bx                         // CHECK-NEXT:     R_X86_64_16
{nf}    orw $foo, %ax, %bx                 // CHECK-NEXT:     R_X86_64_16
{evex}  orw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
{nf}    orw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
orw $foo, 123(%r8,%rax,4), %bx             // CHECK-NEXT:     R_X86_64_16
{nf}    orw $foo, 123(%r8,%rax,4), %bx     // CHECK-NEXT:     R_X86_64_16
{evex}  orl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
{nf}    orl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
orl $foo, %eax, %ebx                       // CHECK-NEXT:     R_X86_64_32
{nf}    orl $foo, %eax, %ebx               // CHECK-NEXT:     R_X86_64_32
{evex}  orl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
{nf}    orl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
orl $foo, 123(%r8,%rax,4), %ebx            // CHECK-NEXT:     R_X86_64_32
{nf}    orl $foo, 123(%r8,%rax,4), %ebx    // CHECK-NEXT:     R_X86_64_32
{evex}  orq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
{nf}    orq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
orq $foo, %rax, %rbx                       // CHECK-NEXT:     R_X86_64_32S
{nf}    orq $foo, %rax, %rbx               // CHECK-NEXT:     R_X86_64_32S
{evex}  orq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
{nf}    orq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
orq $foo, 123(%r8,%rax,4), %rbx            // CHECK-NEXT:     R_X86_64_32S
{nf}    orq $foo, 123(%r8,%rax,4), %rbx    // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
