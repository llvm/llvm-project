# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:      .rela.text {
# CHECK-NEXT:   0x3 R_X86_64_SIZE32 foo 0x0
# CHECK-NEXT:   0xA R_X86_64_SIZE32 foo 0x20
# CHECK-NEXT:   0x11 R_X86_64_SIZE32 foo 0xFFFFFFFFFFFFFFE0
# CHECK-NEXT: }
# CHECK-NEXT: .rela.data {
# CHECK-NEXT:   0x0 R_X86_64_SIZE64 foo 0x0
# CHECK-NEXT:   0x8 R_X86_64_SIZE64 foo 0x20
# CHECK-NEXT:   0x10 R_X86_64_SIZE64 foo 0xFFFFFFFFFFFFFFE0
# CHECK-NEXT: }

movl foo@SIZE, %eax
movl foo@SIZE+32, %eax
movl foo@SIZE-32, %eax

.data
.quad foo@SIZE
.quad foo@SIZE + 32
.quad foo@SIZE - 32
