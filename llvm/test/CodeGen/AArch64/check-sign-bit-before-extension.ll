; RUN: llc -mtriple aarch64-gnu-linux -o -  -asm-verbose=0 %s | FileCheck %s

; These tests make sure that the `cmp` instruction is rendered with an
; instruction that checks the sign bit of the original unextended data
; (%in) instead of the sign bit of the sign extended one that is
; created by the type legalization process.
;
; The tests are subdivided in tests that determine the sign bit
; looking through a `sign_extend_inreg` and tests that determine the
; sign bit looking through a `sign_extend`.

; CHECK-LABEL: f_i8_sign_extend_inreg:
; CHECK: tbnz w0, #7, .LBB
define i32 @f_i8_sign_extend_inreg(i8 %in, i32 %a, i32 %b) nounwind {
entry:
  %cmp = icmp sgt i8 %in, -1
  %ext = zext i8 %in to i32
  br i1 %cmp, label %A, label %B

A:
  %retA = add i32 %ext, %a
  ret i32 %retA

B:
  %retB = add i32 %ext, %b
  ret i32 %retB
}

; CHECK-LABEL: f_i16_sign_extend_inreg:
; CHECK: tbnz w0, #15, .LBB
define i32 @f_i16_sign_extend_inreg(i16 %in, i32 %a, i32 %b) nounwind {
entry:
  %cmp = icmp sgt i16 %in, -1
  %ext = zext i16 %in to i32
  br i1 %cmp, label %A, label %B

A:
  %retA = add i32 %ext, %a
  ret i32 %retA

B:
  %retB = add i32 %ext, %b
  ret i32 %retB
}

; CHECK-LABEL: f_i32_sign_extend_inreg:
; CHECK: tbnz w0, #31, .LBB
define i64 @f_i32_sign_extend_inreg(i32 %in, i64 %a, i64 %b) nounwind {
entry:
  %cmp = icmp sgt i32 %in, -1
  %ext = zext i32 %in to i64
  br i1 %cmp, label %A, label %B

A:
  %retA = add i64 %ext, %a
  ret i64 %retA

B:
  %retB = add i64 %ext, %b
  ret i64 %retB
}

; CHECK-LABEL: g_i8_sign_extend_inreg:
; CHECK: tbnz w0, #7, .LBB
define i32 @g_i8_sign_extend_inreg(i8 %in, i32 %a, i32 %b) nounwind {
entry:
  %cmp = icmp slt i8 %in, 0
  %ext = zext i8 %in to i32
  br i1 %cmp, label %A, label %B

A:
  %retA = add i32 %ext, %a
  ret i32 %retA

B:
  %retB = add i32 %ext, %b
  ret i32 %retB
}

; CHECK-LABEL: g_i16_sign_extend_inreg:
; CHECK: tbnz w0, #15, .LBB
define i32 @g_i16_sign_extend_inreg(i16 %in, i32 %a, i32 %b) nounwind {
entry:
  %cmp = icmp slt i16 %in, 0
  %ext = zext i16 %in to i32
  br i1 %cmp, label %A, label %B

A:
  %retA = add i32 %ext, %a
  ret i32 %retA

B:
  %retB = add i32 %ext, %b
  ret i32 %retB
}

; CHECK-LABEL: g_i32_sign_extend_inreg:
; CHECK: tbnz w0, #31, .LBB
define i64 @g_i32_sign_extend_inreg(i32 %in, i64 %a, i64 %b) nounwind {
entry:
  %cmp = icmp slt i32 %in, 0
  %ext = zext i32 %in to i64
  br i1 %cmp, label %A, label %B

A:
  %retA = add i64 %ext, %a
  ret i64 %retA

B:
  %retB = add i64 %ext, %b
  ret i64 %retB
}

; CHECK-LABEL: f_i32_sign_extend_i64:
; CHECK: tbnz w0, #31, .LBB
define i64 @f_i32_sign_extend_i64(i32 %in, i64 %a, i64 %b) nounwind {
entry:
  %inext = sext i32 %in to i64
  %cmp = icmp sgt i64 %inext, -1
  %ext = zext i32 %in to i64
  br i1 %cmp, label %A, label %B

A:
  %retA = add i64 %ext, %a
  ret i64 %retA

B:
  %retB = add i64 %ext, %b
  ret i64 %retB
}

; CHECK-LABEL: g_i32_sign_extend_i64:
; CHECK: tbnz w0, #31, .LBB
define i64 @g_i32_sign_extend_i64(i32 %in, i64 %a, i64 %b) nounwind {
entry:
  %inext = sext i32 %in to i64
  %cmp = icmp slt i64 %inext, 0
  %ext = zext i32 %in to i64
  br i1 %cmp, label %A, label %B

A:
  %retA = add i64 %ext, %a
  ret i64 %retA

B:
  %retB = add i64 %ext, %b
  ret i64 %retB
}
