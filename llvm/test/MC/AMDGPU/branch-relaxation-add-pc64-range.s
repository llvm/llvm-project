// RUN: llvm-mc -triple=amdgpu12.51 -filetype=obj %s -o %t.o
// RUN: llvm-objdump -d %t.o | FileCheck %s
// RUN: not llvm-mc -triple=amdgpu12.51 -filetype=obj --defsym=ERR=1 %s -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s

// Test the range of branches that are relaxed into s_add_pc_i64. Its 32-bit
// literal is sign-extended, so the offset from the next instruction must fit in
// a signed 32-bit integer. A relaxed s_branch is 8 bytes long, so the offset is
// target - (. + 8). The targets are defined with .set so that they are far away
// without making the object file huge.

// CHECK:      s_add_pc_i64 0x7fffffff
.set fwd_max, . + 0x80000007
  s_branch fwd_max

// The disassembler prints the literal zero-extended.
// CHECK-NEXT: s_add_pc_i64 0x80000000
.set back_max, . - 0x7ffffff8
  s_branch back_max

.ifdef ERR
// ERR: [[@LINE+2]]:12: error: branch size exceeds 32 bits
.set fwd_over, . + 0x80000008
  s_branch fwd_over

// ERR: [[@LINE+2]]:12: error: branch size exceeds 32 bits
.set back_over, . - 0x7ffffff9
  s_branch back_over
.endif
