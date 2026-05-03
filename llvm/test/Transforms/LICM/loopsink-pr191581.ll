; RUN: opt -S -verify-memoryssa -passes=loop-sink < %s | FileCheck %s
; RUN: opt -S -verify-memoryssa -aa-pipeline=basic-aa -passes=loop-sink < %s | FileCheck %s

; Don't sink preheader call with memory access across a later
; conflicting preheader store.
; CHECK:      @PR191581
; CHECK-NEXT: preheader
; CHECK-NEXT: %call = call i32 @n()
; CHECK-NEXT: store i32 9, ptr @i


@i = global i32 0, align 4
@e = global i32 0, align 4

define i32 @n() #0 {
  store i32 1, ptr @i, align 4
  ret i32 0
}

define void @PR191581() !prof !0 {
preheader:
  %call = call i32 @n()
  store i32 9, ptr @i, align 4
  br label %loop_header

loop_header:
  br i1 false, label %loop_inc, label %cold_body

cold_body:
  store i32 %call, ptr @e, align 4
  br label %loop_inc

loop_inc:
  br i1 true, label %exit, label %loop_inc.loop_header_crit_edge, !prof !1

loop_inc.loop_header_crit_edge:
  br label %loop_header

exit:
  ret void
}

attributes #0 = { nounwind memory(write) }

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 -1, i32 0}
