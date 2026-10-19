; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; A per-function V3 promotion stamps every WinEH frame -- the entry frame and
; each EH funclet (each its own .seh_proc) -- with .seh_unwindversion 3.

; Entry frame is V3.
; CHECK-LABEL:  f_eh:
; CHECK:        .seh_proc f_eh
; CHECK:        .seh_unwindversion 3
; CHECK:        .seh_endproc

; The cleanup funclet is a separate .seh_proc and must also be V3.
; CHECK:        .seh_proc "?dtor${{[^"]*}}"
; CHECK:        .seh_unwindversion 3
; CHECK:        .seh_endproc

define dso_local void @f_eh() #0 personality ptr @__C_specific_handler {
entry:
  invoke void @c() to label %ok unwind label %cu
ok:
  ret void
cu:
  %tok = cleanuppad within none []
  call void @d() [ "funclet"(token %tok) ]
  cleanupret from %tok unwind to caller
}

declare void @c()
declare void @d()
declare dso_local i32 @__C_specific_handler(...)
attributes #0 = { uwtable "target-features"="+egpr" }
