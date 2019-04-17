; RUN: opt -S -sccp < %s | FileCheck %s

declare void @BB0_f()
declare void @BB1_f()

; Make sure we can eliminate what is in BB0 as we know that the indirectbr is going to BB1.
;
; CHECK-LABEL: define void @indbrtest1(
; CHECK-NOT: call void @BB0_f()
; CHECK: ret void
define void @indbrtest1() {
entry:
  indirectbr i8* blockaddress(@indbrtest1, %BB1), [label %BB0, label %BB1]
BB0:
  call void @BB0_f()
  br label %BB1
BB1:
  call void @BB1_f()
  ret void
}

; Make sure we can eliminate what is in BB0 as we know that the indirectbr is going to BB1
; by looking through the casts. The casts should be folded away when they are visited
; before the indirectbr instruction.
;
; CHECK-LABEL: define void @indbrtest2(
; CHECK-NOT: call void @BB0_f()
; CHECK: ret void
define void @indbrtest2() {
entry:
  %a = ptrtoint i8* blockaddress(@indbrtest2, %BB1) to i64
  %b = inttoptr i64 %a to i8*
  %c = bitcast i8* %b to i8*
  indirectbr i8* %b, [label %BB0, label %BB1]
BB0:
  call void @BB0_f()
  br label %BB1
BB1:
  call void @BB1_f()
  ret void
}

; Make sure we can not eliminate BB0 as we do not know the target of the indirectbr.
;
; CHECK-LABEL: define void @indbrtest3(
; CHECK: call void @BB0_f()
; CHECK: ret void
define void @indbrtest3(i8** %Q) {
entry:
  %t = load i8*, i8** %Q
  indirectbr i8* %t, [label %BB0, label %BB1]
BB0:
  call void @BB0_f()
  br label %BB1
BB1:
  call void @BB1_f()
  ret void
}

; Make sure we eliminate BB1 as we pick the first successor on undef.
;
; CHECK-LABEL: define void @indbrtest4(
; CHECK: call void @BB0_f()
; CHECK: ret void
define void @indbrtest4(i8** %Q) {
entry:
  indirectbr i8* undef, [label %BB0, label %BB1]
BB0:
  call void @BB0_f()
  br label %BB1
BB1:
  call void @BB1_f()
  ret void
}


