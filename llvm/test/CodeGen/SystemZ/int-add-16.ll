; Test 128-bit addition when the distinct-operands facility is available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Test the case where both operands are in registers.
define i64 @f1(i64 %a, i64 %b, i64 %c, i64 %d, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: algrk %r2, %r4, %r5
; CHECK: alcgr
; CHECK: br %r14
  %x1 = insertelement <2 x i64> undef, i64 %b, i32 0
  %x2 = insertelement <2 x i64> %x1, i64 %c, i32 1
  %x = bitcast <2 x i64> %x2 to i128
  %y2 = insertelement <2 x i64> %x1, i64 %d, i32 1
  %y = bitcast <2 x i64> %y2 to i128
  %add = add i128 %x, %y
  %addv = bitcast i128 %add to <2 x i64>
  %high = extractelement <2 x i64> %addv, i32 0
  store i64 %high, ptr %ptr
  %low = extractelement <2 x i64> %addv, i32 1
  ret i64 %low
}

; Test addition of 1.
define void @f2(i64 %a, i64 %b, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: alghsik {{%r[0-5]}}, %r3, 1
; CHECK: alcgr
; CHECK: br %r14
  %x1 = insertelement <2 x i64> undef, i64 %a, i32 0
  %x2 = insertelement <2 x i64> %x1, i64 %b, i32 1
  %x = bitcast <2 x i64> %x2 to i128
  %add = add i128 %x, 1
  store i128 %add, ptr %ptr
  ret void
}

; Test the upper end of the ALGHSIK range.
define void @f3(i64 %a, i64 %b, ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: alghsik {{%r[0-5]}}, %r3, 32767
; CHECK: alcgr
; CHECK: br %r14
  %x1 = insertelement <2 x i64> undef, i64 %a, i32 0
  %x2 = insertelement <2 x i64> %x1, i64 %b, i32 1
  %x = bitcast <2 x i64> %x2 to i128
  %add = add i128 %x, 32767
  store i128 %add, ptr %ptr
  ret void
}

; Test the next value up, which should use ALGFI instead.
define void @f4(i64 %a, i64 %b, ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: algfi %r3, 32768
; CHECK: alcgr
; CHECK: br %r14
  %x1 = insertelement <2 x i64> undef, i64 %a, i32 0
  %x2 = insertelement <2 x i64> %x1, i64 %b, i32 1
  %x = bitcast <2 x i64> %x2 to i128
  %add = add i128 %x, 32768
  store i128 %add, ptr %ptr
  ret void
}

; Test the lower end of the ALGHSIK range.
define void @f5(i64 %a, i64 %b, ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: alghsik {{%r[0-5]}}, %r3, -32768
; CHECK: alcgr
; CHECK: br %r14
  %x1 = insertelement <2 x i64> undef, i64 %a, i32 0
  %x2 = insertelement <2 x i64> %x1, i64 %b, i32 1
  %x = bitcast <2 x i64> %x2 to i128
  %add = add i128 %x, -32768
  store i128 %add, ptr %ptr
  ret void
}

; Test the next value down, which cannot use either ALGHSIK or ALGFI.
define void @f6(i64 %a, i64 %b, ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK-NOT: alghsik
; CHECK-NOT: algfi
; CHECK: alcgr
; CHECK: br %r14
  %x1 = insertelement <2 x i64> undef, i64 %a, i32 0
  %x2 = insertelement <2 x i64> %x1, i64 %b, i32 1
  %x = bitcast <2 x i64> %x2 to i128
  %add = add i128 %x, -32769
  store i128 %add, ptr %ptr
  ret void
}
