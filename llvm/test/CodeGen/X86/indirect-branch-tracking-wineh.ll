; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; Under CET-IBT (-fcf-protection=branch), WinEH catch/cleanup funclet entries
; are indirect call targets of the C++ EH runtime dispatcher and must begin
; with endbr64, otherwise throwing an exception #CP-faults on an IBT-enforcing
; CPU.

declare i32 @__CxxFrameHandler3(...)
declare void @throws()

define void @f() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: f:
; CHECK: endbr64
entry:
  invoke void @throws() to label %cont unwind label %cd
cont:
  ret void
cd:
  %cs = catchswitch within none [label %c] unwind to caller
c:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %cont
}

; The catch funclet entry must also carry endbr64.
; CHECK: "?catch${{[^"]*}}":
; CHECK: endbr64

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
