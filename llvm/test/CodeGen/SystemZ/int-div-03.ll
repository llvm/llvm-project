; Test 64-bit signed division and remainder when the divisor is
; a signed-extended i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -asm-verbose=0 | FileCheck %s

declare i64 @foo()

; Test register division.  The result is in the second of the two registers.
define void @f1(i64 %dummy, i64 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f1:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgfr %r2, %r4
; CHECK: stg %r3, 0(%r5)
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %div = sdiv i64 %a, %bext
  store i64 %div, ptr %dest
  ret void
}

; Test register remainder.  The result is in the first of the two registers.
define void @f2(i64 %dummy, i64 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f2:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgfr %r2, %r4
; CHECK: stg %r2, 0(%r5)
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %rem = srem i64 %a, %bext
  store i64 %rem, ptr %dest
  ret void
}

; Test that division and remainder use a single instruction.
define i64 @f3(i64 %dummy, i64 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgfr %r2, %r4
; CHECK: ogr %r3, %r2
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %div = sdiv i64 %a, %bext
  %rem = srem i64 %a, %bext
  %or = or i64 %rem, %div
  ret i64 %or
}

; Test register division when the dividend is zero rather than sign extended.
; We can't use dsgfr here
define void @f4(i64 %dummy, i64 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f4:
; CHECK-NOT: dsgfr
; CHECK: br %r14
  %bext = zext i32 %b to i64
  %div = sdiv i64 %a, %bext
  store i64 %div, ptr %dest
  ret void
}

; ...likewise remainder.
define void @f5(i64 %dummy, i64 %a, i32 %b, ptr %dest) {
; CHECK-LABEL: f5:
; CHECK-NOT: dsgfr
; CHECK: br %r14
  %bext = zext i32 %b to i64
  %rem = srem i64 %a, %bext
  store i64 %rem, ptr %dest
  ret void
}

; Test memory division with no displacement.
define void @f6(i64 %dummy, i64 %a, ptr %src, ptr %dest) {
; CHECK-LABEL: f6:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgf %r2, 0(%r4)
; CHECK: stg %r3, 0(%r5)
; CHECK: br %r14
  %b = load i32, ptr %src
  %bext = sext i32 %b to i64
  %div = sdiv i64 %a, %bext
  store i64 %div, ptr %dest
  ret void
}

; Test memory remainder with no displacement.
define void @f7(i64 %dummy, i64 %a, ptr %src, ptr %dest) {
; CHECK-LABEL: f7:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgf %r2, 0(%r4)
; CHECK: stg %r2, 0(%r5)
; CHECK: br %r14
  %b = load i32, ptr %src
  %bext = sext i32 %b to i64
  %rem = srem i64 %a, %bext
  store i64 %rem, ptr %dest
  ret void
}

; Test both memory division and memory remainder.
define i64 @f8(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f8:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgf %r2, 0(%r4)
; CHECK-NOT: {{dsgf|dsgfr}}
; CHECK: ogr %r3, %r2
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %b = load i32, ptr %src
  %bext = sext i32 %b to i64
  %div = sdiv i64 %a, %bext
  %rem = srem i64 %a, %bext
  %or = or i64 %rem, %div
  ret i64 %or
}

; Check the high end of the DSGF range.
define i64 @f9(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: dsgf %r2, 524284(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %rem = srem i64 %a, %bext
  ret i64 %rem
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f10(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f10:
; CHECK: agfi %r4, 524288
; CHECK: dsgf %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %rem = srem i64 %a, %bext
  ret i64 %rem
}

; Check the high end of the negative aligned DSGF range.
define i64 @f11(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f11:
; CHECK: dsgf %r2, -4(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -1
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %rem = srem i64 %a, %bext
  ret i64 %rem
}

; Check the low end of the DSGF range.
define i64 @f12(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f12:
; CHECK: dsgf %r2, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131072
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %rem = srem i64 %a, %bext
  ret i64 %rem
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f13(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f13:
; CHECK: agfi %r4, -524292
; CHECK: dsgf %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131073
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %rem = srem i64 %a, %bext
  ret i64 %rem
}

; Check that DSGF allows an index.
define i64 @f14(i64 %dummy, i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f14:
; CHECK: dsgf %r2, 524287(%r5,%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i32, ptr %ptr
  %bext = sext i32 %b to i64
  %rem = srem i64 %a, %bext
  ret i64 %rem
}

; Make sure that we still use DSGFR rather than DSGR in cases where
; a load and division cannot be combined.
define void @f15(ptr %dest, ptr %src) {
; CHECK-LABEL: f15:
; CHECK: l [[B:%r[0-9]+]], 0(%r3)
; CHECK: brasl %r14, foo@PLT
; CHECK: lgr %r1, %r2
; CHECK: dsgfr %r0, [[B]]
; CHECK: br %r14
  %b = load i32, ptr %src
  %a = call i64 @foo()
  %ext = sext i32 %b to i64
  %div = sdiv i64 %a, %ext
  store i64 %div, ptr %dest
  ret void
}
