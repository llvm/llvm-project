// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  addb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
{nf}    addb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
addb $foo, %al, %bl                         // CHECK-NEXT:     R_X86_64_8
{nf}    addb $foo, %al, %bl                 // CHECK-NEXT:     R_X86_64_8
{evex}  addb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8   
{nf}    addb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8
addb $foo, 123(%r8,%rax,4), %bl             // CHECK-NEXT:     R_X86_64_8
{nf}    addb $foo, 123(%r8,%rax,4), %bl     // CHECK-NEXT:     R_X86_64_8
{evex}  addw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
{nf}    addw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
addw $foo, %ax, %bx                         // CHECK-NEXT:     R_X86_64_16
{nf}    addw $foo, %ax, %bx                 // CHECK-NEXT:     R_X86_64_16
{evex}  addw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
{nf}    addw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
addw $foo, 123(%r8,%rax,4), %bx             // CHECK-NEXT:     R_X86_64_16
{nf}    addw $foo, 123(%r8,%rax,4), %bx     // CHECK-NEXT:     R_X86_64_16
{evex}  addl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
{nf}    addl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
addl $foo, %eax, %ebx                       // CHECK-NEXT:     R_X86_64_32
{nf}    addl $foo, %eax, %ebx               // CHECK-NEXT:     R_X86_64_32
{evex}  addl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
{nf}    addl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
addl $foo, 123(%r8,%rax,4), %ebx            // CHECK-NEXT:     R_X86_64_32
{nf}    addl $foo, 123(%r8,%rax,4), %ebx    // CHECK-NEXT:     R_X86_64_32
{evex}  addq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
{nf}    addq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
addq $foo, %rax, %rbx                       // CHECK-NEXT:     R_X86_64_32S
{nf}    addq $foo, %rax, %rbx               // CHECK-NEXT:     R_X86_64_32S
{evex}  addq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
{nf}    addq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
addq $foo, 123(%r8,%rax,4), %rbx            // CHECK-NEXT:     R_X86_64_32S
{nf}    addq $foo, 123(%r8,%rax,4), %rbx    // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
