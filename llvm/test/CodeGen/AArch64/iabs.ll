; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define i8 @test_i8(i8 %a) nounwind {
; CHECK-LABEL: test_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:  sxtb w8, w0
; CHECK-NEXT:  cmp w8, #0
; CHECK-NEXT:  cneg w0, w8, mi
; CHECK-NEXT:  ret
  %tmp1neg = sub i8 0, %a
  %b = icmp sgt i8 %a, -1
  %abs = select i1 %b, i8 %a, i8 %tmp1neg
  ret i8 %abs
}

define i16 @test_i16(i16 %a) nounwind {
; CHECK-LABEL: test_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:  sxth w8, w0
; CHECK-NEXT:  cmp w8, #0
; CHECK-NEXT:  cneg w0, w8, mi
; CHECK-NEXT:  ret
  %tmp1neg = sub i16 0, %a
  %b = icmp sgt i16 %a, -1
  %abs = select i1 %b, i16 %a, i16 %tmp1neg
  ret i16 %abs
}

define i32 @test_i32(i32 %a) nounwind {
; CHECK-LABEL: test_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:  cmp w0, #0
; CHECK-NEXT:  cneg w0, w0, mi
; CHECK-NEXT:  ret
  %tmp1neg = sub i32 0, %a
  %b = icmp sgt i32 %a, -1
  %abs = select i1 %b, i32 %a, i32 %tmp1neg
  ret i32 %abs
}

define i64 @test_i64(i64 %a) nounwind {
; CHECK-LABEL: test_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:  cmp x0, #0
; CHECK-NEXT:  cneg x0, x0, mi
; CHECK-NEXT:  ret
  %tmp1neg = sub i64 0, %a
  %b = icmp sgt i64 %a, -1
  %abs = select i1 %b, i64 %a, i64 %tmp1neg
  ret i64 %abs
}

