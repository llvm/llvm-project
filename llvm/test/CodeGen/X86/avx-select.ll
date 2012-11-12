; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: _select00
; CHECK: vmovaps
; CHECK-NEXT: LBB
define <8 x i32> @select00(i32 %a, <8 x i32> %b) nounwind {
  %cmpres = icmp eq i32 %a, 255
  %selres = select i1 %cmpres, <8 x i32> zeroinitializer, <8 x i32> %b
  %res = xor <8 x i32> %b, %selres
  ret <8 x i32> %res
}

; CHECK: _select01
; CHECK: vmovaps
; CHECK-NEXT: LBB
define <4 x i64> @select01(i32 %a, <4 x i64> %b) nounwind {
  %cmpres = icmp eq i32 %a, 255
  %selres = select i1 %cmpres, <4 x i64> zeroinitializer, <4 x i64> %b
  %res = xor <4 x i64> %b, %selres
  ret <4 x i64> %res
}

