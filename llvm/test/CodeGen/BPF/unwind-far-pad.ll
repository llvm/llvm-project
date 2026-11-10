; RUN: llc -mtriple=bpfel -mcpu=v4 -gotol-abs-low-bound 0 -filetype=asm -o - %s | FileCheck %s

; unwind_goto has a 16-bit offset like other conditional jumps, and it has to stay
; right after the call.

declare void @may_unwind()
declare void @bpf_free(ptr) nounwind
declare i32 @rust_eh_personality(i32, i64, ptr, ptr)

define i64 @far(ptr %node) uwtable personality ptr @rust_eh_personality {
entry:
  invoke void @may_unwind()
          to label %done unwind label %cleanup

done:
  ret i64 7

cleanup:
  %lp = landingpad { ptr, i32 } cleanup
  call void @bpf_free(ptr %node)
  resume { ptr, i32 } %lp
}

; CHECK-LABEL: far:
; CHECK:         call may_unwind
; CHECK-NEXT:    unwind_goto [[HOP:.LBB[0-9_]+]]
; CHECK:         goto [[DONE:.LBB[0-9_]+]]
; CHECK:       [[HOP]]:
; CHECK-NEXT:    gotol [[PAD:.LBB[0-9_]+]]
; CHECK:       [[DONE]]:
; CHECK-NEXT:    r0 = 7
; CHECK-NEXT:    exit
; CHECK:       [[PAD]]:
; CHECK:         call bpf_free
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    unwind_exit
