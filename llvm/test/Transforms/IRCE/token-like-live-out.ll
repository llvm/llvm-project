; RUN: opt -passes=irce -S < %s | FileCheck %s

; Token-like target extension values cannot be used in PHI nodes. Make sure
; IRCE does not attempt to clone a loop with a live-out resource handle.

define void @target_type_live_out(ptr %resource.ptr) {
; CHECK-LABEL: define void @target_type_live_out(
; CHECK-NOT: preloop
; CHECK-NOT: postloop
entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %resource = load target("dx.RawBuffer", i32, 1, 0), ptr %resource.ptr
  %in.range = icmp slt i32 %idx, 50
  br i1 %in.range, label %in.bounds, label %out.of.bounds

in.bounds:
  %next = icmp slt i32 %idx.next, 2147483647
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
; CHECK: exit:
; CHECK-NEXT: store target("dx.RawBuffer", i32, 1, 0) %resource, ptr %resource.ptr
; CHECK-NEXT: ret void
  store target("dx.RawBuffer", i32, 1, 0) %resource, ptr %resource.ptr
  ret void
}
