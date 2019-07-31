; RUN: opt -S -ipsccp < %s | FileCheck %s

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


; CHECK-LABEL: define internal i32 @indbrtest5(
; CHECK: ret i32 undef
define internal i32 @indbrtest5(i1 %c) {
entry:
  br i1 %c, label %bb1, label %bb2

bb1:
  br label %branch.block


bb2:
  br label %branch.block

branch.block:
  %addr = phi i8* [blockaddress(@indbrtest5, %target1), %bb1], [blockaddress(@indbrtest5, %target2), %bb2]
  indirectbr i8* %addr, [label %target1, label %target2]

target1:
  br label %target2

target2:
  ret i32 10
}


define i32 @indbrtest5_callee(i1 %c) {
; CHECK-LABEL: define i32 @indbrtest5_callee(
; CHECK-NEXT:    %r = call i32 @indbrtest5(i1 %c)
; CHECK-NEXT:    ret i32 10
  %r = call i32 @indbrtest5(i1 %c)
  ret i32 %r
}
