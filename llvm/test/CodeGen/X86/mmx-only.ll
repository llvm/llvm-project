; RUN: llc < %s -mtriple=i686-- -mattr=+mmx | FileCheck %s
; RUN: llc < %s -mtriple=i686-- -mattr=+mmx,-sse | FileCheck %s

; Test that turning off sse doesn't turn off mmx.

declare x86_mmx @llvm.x86.mmx.pcmpgt.d(x86_mmx, x86_mmx) nounwind readnone

define i64 @test88(<1 x i64> %a, <1 x i64> %b) nounwind readnone {
; CHECK-LABEL: @test88
; CHECK: pcmpgtd
entry:
  %0 = bitcast <1 x i64> %b to <2 x i32>
  %1 = bitcast <1 x i64> %a to <2 x i32>
  %mmx_var.i = bitcast <2 x i32> %1 to x86_mmx
  %mmx_var1.i = bitcast <2 x i32> %0 to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.mmx.pcmpgt.d(x86_mmx %mmx_var.i, x86_mmx %mmx_var1.i) nounwind
  %3 = bitcast x86_mmx %2 to <2 x i32>
  %4 = bitcast <2 x i32> %3 to <1 x i64>
  %5 = extractelement <1 x i64> %4, i32 0
  ret i64 %5
}
