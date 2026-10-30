; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs < %s | FileCheck %s
;
; The bool is extended when the call is lowered, that is after the label that
; starts the range of the invoke. It is a select, which takes blocks of its own,
; so the call is not in the block of the label.

declare void @callee(ptr, i64, ptr, i64, i1 zeroext, ptr)
declare void @cleanup() nounwind
declare ptr @get() nounwind
declare i32 @__gcc_personality_v0(...)

define void @f(ptr %v, i64 %n, ptr %x) uwtable personality ptr @__gcc_personality_v0 {
; CHECK-LABEL: f:
; CHECK:       w5 = 1
; CHECK-NEXT:  if r{{[0-9]}} < 65 goto
; CHECK:       w5 = 0
; CHECK-NOT:   r10
; CHECK:       call callee
; CHECK-NEXT:  unwind_goto [[PAD:.LBB0_[0-9]+]]
; CHECK:       exit
; CHECK:       [[PAD]]:
; CHECK:       call cleanup
; CHECK-NEXT:  r0 = 0
; CHECK-NEXT:  unwind_exit
entry:
  %small = icmp ult i64 %n, 1025
  br i1 %small, label %go, label %heap
heap:
  %h = call ptr @get()
  br label %go
go:
  %p = phi ptr [ %v, %entry ], [ %h, %heap ]
  %c = icmp ult i64 %n, 65
  invoke void @callee(ptr %v, i64 %n, ptr %p, i64 %n, i1 zeroext %c, ptr %x)
          to label %ok unwind label %pad
ok:
  ret void
pad:
  %lp = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %lp
}
