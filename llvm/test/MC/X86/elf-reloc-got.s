# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readobj -r  - | FileCheck  %s

# CHECK:      .rela.GOT64 {
# CHECK-NEXT:   0x2 R_X86_64_GOT64 dat 0x0
# CHECK-NEXT:   0xC R_X86_64_GOT64 und 0x0
# CHECK-NEXT:   0x16 R_X86_64_GOT64 .GOT64 0x0
# CHECK-NEXT: }

.section .GOT64,"ax"
movabs $dat@GOT, %rax
movabs $und@GOT, %rax
movabs $.GOT64@GOT, %rax

.data
dat:
