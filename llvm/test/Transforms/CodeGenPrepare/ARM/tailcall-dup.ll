; RUN: opt -codegenprepare -S < %s | FileCheck %s

target triple = "armv8m.main-none-eabi"

declare i8* @f0()
declare i8* @f1()

define i8* @tail_dup() {
; CHECK-LABEL: tail_dup
; CHECK: tail call i8* @f0()
; CHECK-NEXT: ret i8*
; CHECK: tail call i8* @f1()
; CHECK-NEXT: ret i8*
bb0:
  %tmp0 = tail call i8* @f0()
  br label %return
bb1:
  %tmp1 = tail call i8* @f1()
  br label %return
return:
  %retval = phi i8* [ %tmp0, %bb0 ], [ %tmp1, %bb1 ]
  ret i8* %retval
}

define nonnull i8* @nonnull_dup() {
; CHECK-LABEL: nonnull_dup
; CHECK: tail call i8* @f0()
; CHECK-NEXT: ret i8*
; CHECK: tail call i8* @f1()
; CHECK-NEXT: ret i8*
bb0:
  %tmp0 = tail call i8* @f0()
  br label %return
bb1:
  %tmp1 = tail call i8* @f1()
  br label %return
return:
  %retval = phi i8* [ %tmp0, %bb0 ], [ %tmp1, %bb1 ]
  ret i8* %retval
}

define i8* @noalias_dup() {
; CHECK-LABEL: noalias_dup
; CHECK: tail call noalias i8* @f0()
; CHECK-NEXT: ret i8*
; CHECK: tail call noalias i8* @f1()
; CHECK-NEXT: ret i8*
bb0:
  %tmp0 = tail call noalias i8* @f0()
  br label %return
bb1:
  %tmp1 = tail call noalias i8* @f1()
  br label %return
return:
  %retval = phi i8* [ %tmp0, %bb0 ], [ %tmp1, %bb1 ]
  ret i8* %retval
}

; Use inreg as a way of testing that attributes (other than nonnull and
; noalias) disable the tailcall duplication in cgp.

define inreg i8* @inreg_nodup() {
; CHECK-LABEL: inreg_nodup
; CHECK: tail call i8* @f0()
; CHECK-NEXT: br label %return
; CHECK: tail call i8* @f1()
; CHECK-NEXT: br label %return
bb0:
  %tmp0 = tail call i8* @f0()
  br label %return
bb1:
  %tmp1 = tail call i8* @f1()
  br label %return
return:
  %retval = phi i8* [ %tmp0, %bb0 ], [ %tmp1, %bb1 ]
  ret i8* %retval
}
