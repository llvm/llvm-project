; Testg 64-bit unsigned division and remainder.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -asm-verbose=0 | FileCheck %s

declare i64 @foo()

; Testg register division.  The result is in the second of the two registers.
define void @f1(i64 %dummy, i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlgr %r2, %r4
; CHECK: stg %r3, 0(%r5)
; CHECK: br %r14
  %div = udiv i64 %a, %b
  store i64 %div, ptr %dest
  ret void
}

; Testg register remainder.  The result is in the first of the two registers.
define void @f2(i64 %dummy, i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlgr %r2, %r4
; CHECK: stg %r2, 0(%r5)
; CHECK: br %r14
  %rem = urem i64 %a, %b
  store i64 %rem, ptr %dest
  ret void
}

; Testg that division and remainder use a single instruction.
define i64 @f3(i64 %dummy1, i64 %a, i64 %b) {
; CHECK-LABEL: f3:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlgr %r2, %r4
; CHECK-NOT: dlgr
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %div = udiv i64 %a, %b
  %rem = urem i64 %a, %b
  %or = or i64 %rem, %div
  ret i64 %or
}

; Testg memory division with no displacement.
define void @f4(i64 %dummy, i64 %a, ptr %src, ptr %dest) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlg %r2, 0(%r4)
; CHECK: stg %r3, 0(%r5)
; CHECK: br %r14
  %b = load i64, ptr %src
  %div = udiv i64 %a, %b
  store i64 %div, ptr %dest
  ret void
}

; Testg memory remainder with no displacement.
define void @f5(i64 %dummy, i64 %a, ptr %src, ptr %dest) {
; CHECK-LABEL: f5:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlg %r2, 0(%r4)
; CHECK: stg %r2, 0(%r5)
; CHECK: br %r14
  %b = load i64, ptr %src
  %rem = urem i64 %a, %b
  store i64 %rem, ptr %dest
  ret void
}

; Testg both memory division and memory remainder.
define i64 @f6(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f6:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlg %r2, 0(%r4)
; CHECK-NOT: {{dlg|dlgr}}
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %b = load i64, ptr %src
  %div = udiv i64 %a, %b
  %rem = urem i64 %a, %b
  %or = or i64 %rem, %div
  ret i64 %or
}

; Check the high end of the DLG range.
define i64 @f7(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: dlg %r2, 524280(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65535
  %b = load i64, ptr %ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f8(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r4, 524288
; CHECK: dlg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65536
  %b = load i64, ptr %ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check the high end of the negative aligned DLG range.
define i64 @f9(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: dlg %r2, -8(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -1
  %b = load i64, ptr %ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check the low end of the DLG range.
define i64 @f10(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f10:
; CHECK: dlg %r2, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65536
  %b = load i64, ptr %ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f11(i64 %dummy, i64 %a, ptr %src) {
; CHECK-LABEL: f11:
; CHECK: agfi %r4, -524296
; CHECK: dlg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65537
  %b = load i64, ptr %ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check that DLG allows an index.
define i64 @f12(i64 %dummy, i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f12:
; CHECK: dlg %r2, 524287(%r5,%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i64, ptr %ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check that divisions of spilled values can use DLG rather than DLGR.
define i64 @f13(ptr %ptr0) {
; CHECK-LABEL: f13:
; CHECK: brasl %r14, foo@PLT
; CHECK: dlg {{%r[0-9]+}}, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i64, ptr %ptr0, i64 2
  %ptr2 = getelementptr i64, ptr %ptr0, i64 4
  %ptr3 = getelementptr i64, ptr %ptr0, i64 6
  %ptr4 = getelementptr i64, ptr %ptr0, i64 8
  %ptr5 = getelementptr i64, ptr %ptr0, i64 10
  %ptr6 = getelementptr i64, ptr %ptr0, i64 12
  %ptr7 = getelementptr i64, ptr %ptr0, i64 14
  %ptr8 = getelementptr i64, ptr %ptr0, i64 16
  %ptr9 = getelementptr i64, ptr %ptr0, i64 18
  %ptr10 = getelementptr i64, ptr %ptr0, i64 20

  %val0 = load i64, ptr %ptr0
  %val1 = load i64, ptr %ptr1
  %val2 = load i64, ptr %ptr2
  %val3 = load i64, ptr %ptr3
  %val4 = load i64, ptr %ptr4
  %val5 = load i64, ptr %ptr5
  %val6 = load i64, ptr %ptr6
  %val7 = load i64, ptr %ptr7
  %val8 = load i64, ptr %ptr8
  %val9 = load i64, ptr %ptr9
  %val10 = load i64, ptr %ptr10

  %ret = call i64 @foo()

  %div0 = udiv i64 %ret, %val0
  %div1 = udiv i64 %div0, %val1
  %div2 = udiv i64 %div1, %val2
  %div3 = udiv i64 %div2, %val3
  %div4 = udiv i64 %div3, %val4
  %div5 = udiv i64 %div4, %val5
  %div6 = udiv i64 %div5, %val6
  %div7 = udiv i64 %div6, %val7
  %div8 = udiv i64 %div7, %val8
  %div9 = udiv i64 %div8, %val9
  %div10 = udiv i64 %div9, %val10

  ret i64 %div10
}
