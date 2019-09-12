; RUN: llc < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s
; RUN: llc -global-isel < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

@t = weak global i32 ()* null
@x = external global i32, align 4

define void @t2() {
; CHECK-LABEL: t2:
; CHECK: adrp	x[[GOTADDR:[0-9]+]], _t@GOTPAGE
; CHECK: ldr	x[[ADDR:[0-9]+]], [x[[GOTADDR]], _t@GOTPAGEOFF]
; CHECK: ldr	x[[DEST:[0-9]+]], [x[[ADDR]]]
; CHECK: br	x[[DEST]]
  %tmp = load i32 ()*, i32 ()** @t
  %tmp.upgrd.2 = tail call i32 %tmp()
  ret void
}

define void @t3() {
; CHECK-LABEL: t3:
; CHECK: b	_t2
  tail call void @t2()
  ret void
}

define double @t4(double %a) nounwind readonly ssp {
; CHECK-LABEL: t4:
; CHECK: b	_sin
  %tmp = tail call double @sin(double %a) nounwind readonly
  ret double %tmp
}

define float @t5(float %a) nounwind readonly ssp {
; CHECK-LABEL: t5:
; CHECK: b	_sinf
  %tmp = tail call float @sinf(float %a) nounwind readonly
  ret float %tmp
}

define void @t7() nounwind {
; CHECK-LABEL: t7:
; CHECK: b	_foo
; CHECK: b	_bar

  br i1 undef, label %bb, label %bb1.lr.ph

bb1.lr.ph:                                        ; preds = %entry
  tail call void @bar() nounwind
  ret void

bb:                                               ; preds = %entry
  tail call void @foo() nounwind
  ret void
}

define i32 @t8(i32 %x) nounwind ssp {
; CHECK-LABEL: t8:
; CHECK: b	_c
; CHECK: b	_a
; CHECK: b	_b
  %and = and i32 %x, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call i32 @a(i32 %x) nounwind
  br label %return

if.end:                                           ; preds = %entry
  %and1 = and i32 %x, 2
  %tobool2 = icmp eq i32 %and1, 0
  br i1 %tobool2, label %if.end5, label %if.then3

if.then3:                                         ; preds = %if.end
  %call4 = tail call i32 @b(i32 %x) nounwind
  br label %return

if.end5:                                          ; preds = %if.end
  %call6 = tail call i32 @c(i32 %x) nounwind
  br label %return

return:                                           ; preds = %if.end5, %if.then3, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call4, %if.then3 ], [ %call6, %if.end5 ]
  ret i32 %retval.0
}

declare float @sinf(float) nounwind readonly
declare double @sin(double) nounwind readonly
declare void @bar() nounwind
declare void @foo() nounwind
declare i32 @a(i32)
declare i32 @b(i32)
declare i32 @c(i32)

; CHECK-LABEL: tswift:
; CHECK: b _swiftfunc
define swiftcc i32 @tswift(i32 %a) nounwind {
  %res = tail call i32 @swiftfunc(i32 %a)
  ret i32 %res
}

declare swiftcc i32 @swiftfunc(i32) nounwind
