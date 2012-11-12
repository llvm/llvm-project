; RUN: llc < %s -march=x86 -mcpu=corei7-avx -mattr=+avx -mtriple=i686-pc-win32 | FileCheck %s

; CHECK: endless_loop
define void @endless_loop() {
entry:
  %0 = load <8 x i32> addrspace(1)* undef, align 32
  %1 = shufflevector <8 x i32> %0, <8 x i32> undef, <16 x i32> <i32 4, i32 4, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = shufflevector <16 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 undef>, <16 x i32> %1, <16 x i32> <i32 16, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 17>
  store <16 x i32> %2, <16 x i32> addrspace(1)* undef, align 64
  ret void
; CHECK: ret
}
