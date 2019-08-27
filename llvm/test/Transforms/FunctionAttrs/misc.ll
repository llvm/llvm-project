; RUN: opt -S -attributor -attributor-disable=false < %s | FileCheck %s

define void @external() {
entry:
  %a = alloca i32, align 4
  %tmp = bitcast i32* %a to i8*
  call void @foo(i32* nonnull %a)
; Check we do not crash on these uses
; CHECK: call void @callback1(void (i32*)* nonnull @foo)
  call void @callback1(void (i32*)* nonnull @foo)
; CHECK: call void @callback2(void (i8*)* nonnull bitcast (void (i32*)* @foo to void (i8*)*))
  call void @callback2(void (i8*)* bitcast (void (i32*)* @foo to void (i8*)*))
  %tmp1 = bitcast i32* %a to i8*
  ret void
}

define internal void @foo(i32* %a) {
entry:
  ret void
}

declare void @callback1(void (i32*)*)
declare void @callback2(void (i8*)*)
