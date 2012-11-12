; RUN: llc -asm-verbose -mtriple=x86_64-unknown-unknown -mcpu=corei7 < %s | FileCheck %s
define <16 x i8> @shuf(<16 x i8> %inval1) {
entry:
  %0 = shufflevector <16 x i8> %inval1, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 4, i32 3, i32 2, i32 16, i32 16, i32 3, i32 4, i32 0, i32 4, i32 3, i32 2, i32 16, i32 16, i32 3, i32 4>
  ret <16 x i8> %0
; CHECK: shuf
; CHECK: # BB#0: # %entry
; CHECK-NEXT: pshufb
; CHECK-NEXT: ret
}
