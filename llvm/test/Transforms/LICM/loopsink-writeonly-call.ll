; RUN: opt -S -verify-memoryssa -passes=loop-sink < %s | FileCheck %s

; A call that writes memory is observable even when its return value is only
; used on a cold path through the loop. It must not be sunk from the preheader.
;
; CHECK-LABEL: define void @writeonly_call(
; CHECK:       entry:
; CHECK-NEXT:    [[VALUE:%.*]] = call i32 @write()
; CHECK-NEXT:    br label %loop
; CHECK:       cold:
; CHECK-NOT:     call i32 @write()
define void @writeonly_call(i1 %condition) !prof !0 {
entry:
  %value = call i32 @write()
  br label %loop

loop:
  br i1 %condition, label %cold, label %exit, !prof !1

cold:
  %unused = icmp eq i32 %value, 0
  br label %loop

exit:
  ret void
}

declare i32 @write() nounwind willreturn memory(write)

; A call without willreturn cannot move to a path that may not execute.
;
; CHECK-LABEL: define void @call_without_willreturn(
; CHECK:       entry:
; CHECK-NEXT:    [[VALUE:%.*]] = call i32 @may_not_return()
; CHECK-NEXT:    br label %loop
; CHECK:       cold:
; CHECK-NOT:     call i32 @may_not_return()
define void @call_without_willreturn(i1 %condition) !prof !0 {
entry:
  %value = call i32 @may_not_return()
  br label %loop

loop:
  br i1 %condition, label %cold, label %exit, !prof !1

cold:
  %unused = icmp eq i32 %value, 0
  br label %loop

exit:
  ret void
}

declare i32 @may_not_return() nounwind memory(none)

; A side-effect-free call that is guaranteed to return remains sinkable.
;
; CHECK-LABEL: define void @readonly_willreturn_call(
; CHECK:       entry:
; CHECK-NEXT:    br label %loop
; CHECK:       cold:
; CHECK-NEXT:    [[VALUE:%.*]] = call i32 @read()
define void @readonly_willreturn_call(i1 %condition) !prof !0 {
entry:
  %value = call i32 @read()
  br label %loop

loop:
  br i1 %condition, label %cold, label %exit, !prof !1

cold:
  %unused = icmp eq i32 %value, 0
  br label %loop

exit:
  ret void
}

declare i32 @read() nounwind willreturn memory(read)

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 1, i32 2000}
