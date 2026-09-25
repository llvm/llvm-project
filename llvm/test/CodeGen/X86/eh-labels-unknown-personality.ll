; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Label suppression is keyed on the target's EH-table format, not on
; recognizing the personality, so an unknown personality still gets
; its per-invoke labels.

; CHECK-LABEL: f:
; CHECK:      .Ltmp0:
; CHECK-NEXT:   callq g@PLT
; CHECK-NEXT: .Ltmp1:
; CHECK:      .gcc_except_table
; CHECK:      .uleb128 .Ltmp0-.Lfunc_begin0
define void @f() personality ptr @custom_personality {
entry:
  invoke void @g()
          to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp
}

declare void @g()
declare i32 @custom_personality(...)
