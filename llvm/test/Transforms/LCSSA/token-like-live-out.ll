; RUN: opt -passes='lcssa,verify' -verify-loop-info -S < %s | FileCheck %s

; Token-like target extension values cannot be used in PHI nodes. Make sure
; LCSSA formation leaves a live-out resource handle unchanged.

define void @target_type_live_out(ptr %resource.ptr) {
; CHECK-LABEL: define void @target_type_live_out(
entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %loop ]
  %resource = load target("dx.RawBuffer", i32, 1, 0), ptr %resource.ptr
  %idx.next = add i32 %idx, 1
  %continue = icmp slt i32 %idx.next, 100
  br i1 %continue, label %loop, label %exit

exit:
; CHECK: exit:
; CHECK-NEXT: store target("dx.RawBuffer", i32, 1, 0) %resource, ptr %resource.ptr
; CHECK-NEXT: ret void
  store target("dx.RawBuffer", i32, 1, 0) %resource, ptr %resource.ptr
  ret void
}
