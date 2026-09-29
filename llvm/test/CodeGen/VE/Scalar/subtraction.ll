; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func8s(i8 signext %0, i8 signext %1) {
; CHECK-LABEL: func8s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub i8 %0, %1
  ret i8 %3
}

define signext i16 @func16s(i16 signext %0, i16 signext %1) {
; CHECK-LABEL: func16s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub i16 %0, %1
  ret i16 %3
}

define signext i32 @func32s(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: func32s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub nsw i32 %0, %1
  ret i32 %3
}

define i64 @func64s(i64 %0, i64 %1) {
; CHECK-LABEL: func64s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub nsw i64 %0, %1
  ret i64 %3
}

define i128 @func128s(i128 %0, i128 %1) {
; CHECK-LABEL: func128s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.l %s1, %s1, %s3
; CHECK-NEXT:    cmpu.l %s3, %s0, %s2
; CHECK-NEXT:    or %s4, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s4, (63)0, %s3
; CHECK-NEXT:    adds.w.zx %s3, %s4, (0)1
; CHECK-NEXT:    subs.l %s1, %s1, %s3
; CHECK-NEXT:    subs.l %s0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub nsw i128 %0, %1
  ret i128 %3
}

define zeroext i8 @func8z(i8 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: func8z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub i8 %0, %1
  ret i8 %3
}

define zeroext i16 @func16z(i16 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: func16z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub i16 %0, %1
  ret i16 %3
}

define zeroext i32 @func32z(i32 zeroext %0, i32 zeroext %1) {
; CHECK-LABEL: func32z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub i32 %0, %1
  ret i32 %3
}

define i64 @func64z(i64 %0, i64 %1) {
; CHECK-LABEL: func64z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub i64 %0, %1
  ret i64 %3
}

define i128 @func128z(i128 %0, i128 %1) {
; CHECK-LABEL: func128z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subs.l %s1, %s1, %s3
; CHECK-NEXT:    cmpu.l %s3, %s0, %s2
; CHECK-NEXT:    or %s4, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s4, (63)0, %s3
; CHECK-NEXT:    adds.w.zx %s3, %s4, (0)1
; CHECK-NEXT:    subs.l %s1, %s1, %s3
; CHECK-NEXT:    subs.l %s0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sub i128 %0, %1
  ret i128 %3
}

define signext i8 @funci8s(i8 signext %a) {
; CHECK-LABEL: funci8s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, -5, %s0
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add i8 %a, -5
  ret i8 %ret
}

define signext i16 @funci16s(i16 signext %a) {
; CHECK-LABEL: funci16s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, -5, %s0
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add i16 %a, -5
  ret i16 %ret
}

define signext i32 @funci32s(i32 signext %a) {
; CHECK-LABEL: funci32s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, -5, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add i32 %a, -5
  ret i32 %ret
}

define i64 @funci64s(i64 %a) {
; CHECK-LABEL: funci64s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -5(, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add nsw i64 %a, -5
  ret i64 %ret
}

define i128 @funci128s(i128 %0) {
; CHECK-LABEL: funci128s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, -5(, %s0)
; CHECK-NEXT:    cmpu.l %s0, %s2, %s0
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    lea %s1, -1(%s0, %s1)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = add nsw i128 %0, -5
  ret i128 %2
}

define zeroext i8 @funci8z(i8 zeroext %a) {
; CHECK-LABEL: funci8z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, -5, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add i8 %a, -5
  ret i8 %ret
}

define zeroext i16 @funci16z(i16 zeroext %a) {
; CHECK-LABEL: funci16z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, -5, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add i16 %a, -5
  ret i16 %ret
}

define zeroext i32 @funci32z(i32 zeroext %a) {
; CHECK-LABEL: funci32z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, -5, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add i32 %a, -5
  ret i32 %ret
}

define i64 @funci64z(i64 %a) {
; CHECK-LABEL: funci64z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -5(, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add i64 %a, -5
  ret i64 %ret
}

define i128 @funci128z(i128 %0) {
; CHECK-LABEL: funci128z:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, -5(, %s0)
; CHECK-NEXT:    cmpu.l %s0, %s2, %s0
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    lea %s1, -1(%s0, %s1)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = add i128 %0, -5
  ret i128 %2
}

define i64 @funci64_2(i64 %a) {
; CHECK-LABEL: funci64_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -2147483648(, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = add nsw i64 %a, -2147483648
  ret i64 %ret
}

define i128 @funci128_2(i128 %0) {
; CHECK-LABEL: funci128_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, -2147483648(, %s0)
; CHECK-NEXT:    cmpu.l %s0, %s2, %s0
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    lea %s1, -1(%s0, %s1)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = add nsw i128 %0, -2147483648
  ret i128 %2
}
