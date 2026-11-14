; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -win-eh-prepare < %s | FileCheck %s
; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)
declare void @f()
declare void @h(i32)

; Both the invoke argument and the PHI on its normal successor use the current
; value of the EH PHI. Reload it before the spill store for the next unwind
; overwrites its slot, even though both uses occur at the end of reenter.
; CHECK-LABEL: @snapshot_before_unwind_store(
define void @snapshot_before_unwind_store() personality ptr @__CxxFrameHandler3 {
entry:
  ; CHECK: entry:
  ; CHECK: [[Slot:%[^ ]+]] = alloca i32
  ; CHECK-NEXT: store i32 0, ptr [[Slot]]
  ; CHECK-NEXT: invoke void @f()
  invoke void @f()
          to label %normal unwind label %dispatch

dispatch:
  ; CHECK: dispatch:
  ; CHECK-NEXT: %cs = catchswitch
  %old = phi i32 [ 0, %entry ], [ 1, %reenter ]
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %reenter

reenter:
  ; CHECK: reenter:
  ; CHECK-NEXT: load i32, ptr [[Slot]]
  ; CHECK-NEXT: load i32, ptr [[Slot]]
  ; CHECK-NEXT: store i32 1, ptr [[Slot]]
  ; CHECK-NEXT: invoke void @h(i32 %{{old.wineh.reload[0-9]*}})
  invoke void @h(i32 %old)
          to label %normal unwind label %dispatch

normal:
  ; CHECK: normal:
  ; CHECK-NEXT: %result = phi i32 [ 7, %entry ], [ %{{old.wineh.reload[0-9]*}}, %reenter ]
  ; CHECK-NEXT: call void @h(i32 %result)
  %result = phi i32 [ 7, %entry ], [ %old, %reenter ]
  call void @h(i32 %result)
  ret void
}
