; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

define i32 @testsize1(i32 %x) minsize nounwind {
entry:
       %div = sdiv i32 %x, 32
       ret i32 %div
; CHECK-LABEL: testsize1
; CHECK: sdiv 
}

define i32 @testsize2(i32 %x) minsize nounwind {
entry:
       %div = sdiv i32 %x, 33
       ret i32 %div
; CHECK-LABEL: testsize2
; CHECK: sdiv
}

define i32 @testsize3(i32 %x) minsize nounwind {
entry:
       %div = udiv i32 %x, 32
       ret i32 %div
; CHECK-LABEL: testsize3
; CHECK: lsr
}

define i32 @testsize4(i32 %x) minsize nounwind {
entry:
       %div = udiv i32 %x, 33
       ret i32 %div
; CHECK-LABEL: testsize4
; CHECK: udiv 
}

define <8 x i16> @sdiv_vec8x16_minsize(<8 x i16> %var) minsize {
entry:
; CHECK: sdiv_vec8x16_minsize
; CHECK: sshr 	v1.8h, v0.8h, #15 
; CHECK: usra	v0.8h, v1.8h, #11
; CHECK: sshr	v0.8h, v0.8h, #5
; CHECK: ret
  %0 = sdiv <8 x i16> %var, <i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32>
  ret <8 x i16> %0
}

