// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
{evex}  subb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
{nf}    subb $foo, %al                      // CHECK-NEXT:     R_X86_64_8
subb $foo, %al, %bl                         // CHECK-NEXT:     R_X86_64_8
{nf}    subb $foo, %al, %bl                 // CHECK-NEXT:     R_X86_64_8
{evex}  subb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8   
{nf}    subb $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_8
subb $foo, 123(%r8,%rax,4), %bl             // CHECK-NEXT:     R_X86_64_8
{nf}    subb $foo, 123(%r8,%rax,4), %bl     // CHECK-NEXT:     R_X86_64_8
{evex}  subw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
{nf}    subw $foo, %ax                      // CHECK-NEXT:     R_X86_64_16
subw $foo, %ax, %bx                         // CHECK-NEXT:     R_X86_64_16
{nf}    subw $foo, %ax, %bx                 // CHECK-NEXT:     R_X86_64_16
{evex}  subw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
{nf}    subw $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_16
subw $foo, 123(%r8,%rax,4), %bx             // CHECK-NEXT:     R_X86_64_16
{nf}    subw $foo, 123(%r8,%rax,4), %bx     // CHECK-NEXT:     R_X86_64_16
{evex}  subl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
{nf}    subl $foo, %eax                     // CHECK-NEXT:     R_X86_64_32
subl $foo, %eax, %ebx                       // CHECK-NEXT:     R_X86_64_32
{nf}    subl $foo, %eax, %ebx               // CHECK-NEXT:     R_X86_64_32
{evex}  subl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
{nf}    subl $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32
subl $foo, 123(%r8,%rax,4), %ebx            // CHECK-NEXT:     R_X86_64_32
{nf}    subl $foo, 123(%r8,%rax,4), %ebx    // CHECK-NEXT:     R_X86_64_32
{evex}  subq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
{nf}    subq $foo, %rax                     // CHECK-NEXT:     R_X86_64_32S
subq $foo, %rax, %rbx                       // CHECK-NEXT:     R_X86_64_32S
{nf}    subq $foo, %rax, %rbx               // CHECK-NEXT:     R_X86_64_32S
{evex}  subq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
{nf}    subq $foo, 123(%r8,%rax,4)          // CHECK-NEXT:     R_X86_64_32S
subq $foo, 123(%r8,%rax,4), %rbx            // CHECK-NEXT:     R_X86_64_32S
{nf}    subq $foo, 123(%r8,%rax,4), %rbx    // CHECK-NEXT:     R_X86_64_32S
// CHECK-NEXT:   }
// CHECK-NEXT: ]
