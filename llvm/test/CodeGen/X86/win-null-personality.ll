; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; A personality that is not a function leaves nothing to emit a symbol for, so
; no personality or handler data should be produced. This used to crash the
; X86 assembly printer.

; CHECK-LABEL: null_personality:
; CHECK-NOT: .seh_handler
; CHECK: retq

define void @null_personality() personality ptr null {
  ret void
}

; A real personality with a landing pad still emits its handler as before.

; CHECK-LABEL: real_personality:
; CHECK: .seh_handler __CxxFrameHandler3

declare void @g()
declare i32 @__CxxFrameHandler3(...)

define void @real_personality() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %cont unwind label %catch

cont:
  ret void

catch:
  %cs = catchswitch within none [label %handler] unwind to caller

handler:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %cont
}
