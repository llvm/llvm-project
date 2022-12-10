; RUN: opt -codegenprepare -S < %s | FileCheck %s

target triple = "armv8m.main-none-eabi"

declare ptr @f0()
declare ptr @f1()
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind

define ptr @tail_dup() {
; CHECK-LABEL: tail_dup
; CHECK: tail call ptr @f0()
; CHECK-NEXT: ret ptr
; CHECK: tail call ptr @f1()
; CHECK-NEXT: ret ptr
bb0:
  %a = alloca i32
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a) nounwind
  %tmp0 = tail call ptr @f0()
  br label %return
bb1:
  %tmp1 = tail call ptr @f1()
  br label %return
return:
  %retval = phi ptr [ %tmp0, %bb0 ], [ %tmp1, %bb1 ]
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a) nounwind
  ret ptr %retval
}

define nonnull ptr @nonnull_dup() {
; CHECK-LABEL: nonnull_dup
; CHECK: tail call ptr @f0()
; CHECK-NEXT: ret ptr
; CHECK: tail call ptr @f1()
; CHECK-NEXT: ret ptr
bb0:
  %tmp0 = tail call ptr @f0()
  br label %return
bb1:
  %tmp1 = tail call ptr @f1()
  br label %return
return:
  %retval = phi ptr [ %tmp0, %bb0 ], [ %tmp1, %bb1 ]
  ret ptr %retval
}

define ptr @noalias_dup() {
; CHECK-LABEL: noalias_dup
; CHECK: tail call noalias ptr @f0()
; CHECK-NEXT: ret ptr
; CHECK: tail call noalias ptr @f1()
; CHECK-NEXT: ret ptr
bb0:
  %tmp0 = tail call noalias ptr @f0()
  br label %return
bb1:
  %tmp1 = tail call noalias ptr @f1()
  br label %return
return:
  %retval = phi ptr [ %tmp0, %bb0 ], [ %tmp1, %bb1 ]
  ret ptr %retval
}

; Use inreg as a way of testing that attributes (other than nonnull and
; noalias) disable the tailcall duplication in cgp.

define inreg ptr @inreg_nodup() {
; CHECK-LABEL: inreg_nodup
; CHECK: tail call ptr @f0()
; CHECK-NEXT: br label %return
; CHECK: tail call ptr @f1()
; CHECK-NEXT: br label %return
bb0:
  %tmp0 = tail call ptr @f0()
  br label %return
bb1:
  %tmp1 = tail call ptr @f1()
  br label %return
return:
  %retval = phi ptr [ %tmp0, %bb0 ], [ %tmp1, %bb1 ]
  ret ptr %retval
}
