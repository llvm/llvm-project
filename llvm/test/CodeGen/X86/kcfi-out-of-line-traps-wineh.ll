; RUN: llc -mtriple=x86_64-pc-windows-msvc -verify-machineinstrs < %s | FileCheck %s

;; Windows EH (C++ EH and SEH) functions fall back to inline KCFI checks so
;; faultable checks stay inside guarded ranges without extra EH state
;; bookkeeping.

declare i32 @__CxxFrameHandler3(...)
declare i32 @__C_specific_handler(...)
declare void @cleanup()

define void @win_invoke(ptr noundef %x) personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: win_invoke:
; CHECK:       [[BEGIN:\.Ltmp[0-9]+]]: # EH_LABEL
; CHECK-NEXT:    movl $4282621618, %r10d # imm = 0xFF439EB2
; CHECK-NEXT:    addl -4(%rcx), %r10d
; CHECK-NEXT:    je [[PASS:\.Ltmp[0-9]+]]
; CHECK:         ud2
; CHECK-NEXT:  [[PASS]]:
; CHECK-NEXT:    callq *%rcx
; CHECK-NEXT:    nop
; CHECK-NEXT:  [[END:\.Ltmp[0-9]+]]: # EH_LABEL
; CHECK:       $ip2state$win_invoke:
; CHECK:         .long [[BEGIN]]@IMGREL # IP
; CHECK-NEXT:    .long 0 # ToState
; CHECK-NEXT:    .long [[END]]@IMGREL # IP
; CHECK-NEXT:    .long -1 # ToState
entry:
  invoke void %x() [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %ehcleanup
cont:
  ret void
ehcleanup:
  %cp = cleanuppad within none []
  call void @cleanup() [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

define i32 @win_seh(ptr noundef %x) personality ptr @__C_specific_handler {
; CHECK-LABEL: win_seh:
; CHECK:       [[BEGIN:\.Ltmp[0-9]+]]: # EH_LABEL
; CHECK-NEXT:    movl $4282621618, %r10d # imm = 0xFF439EB2
; CHECK-NEXT:    addl -4(%rcx), %r10d
; CHECK-NEXT:    je [[PASS:\.Ltmp[0-9]+]]
; CHECK:         ud2
; CHECK-NEXT:  [[PASS]]:
; CHECK-NEXT:    callq *%rcx
; CHECK-NEXT:    nop
; CHECK-NEXT:  [[END:\.Ltmp[0-9]+]]: # EH_LABEL
; CHECK:         .long [[BEGIN]]@IMGREL # LabelStart
; CHECK-NEXT:    .long [[END]]@IMGREL # LabelEnd
entry:
  invoke void %x() [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %catch.dispatch
catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs [ptr null]
  catchret from %cp to label %cont
cont:
  ret i32 0
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 8, !"kcfi-out-of-line-traps", i32 1}
