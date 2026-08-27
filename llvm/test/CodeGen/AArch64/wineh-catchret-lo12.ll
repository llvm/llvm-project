; RUN: llc -mtriple=aarch64-pc-windows-msvc < %s | FileCheck %s
; The output has to assemble as well as compile:
; RUN: llc -mtriple=aarch64-pc-windows-msvc < %s | llvm-mc -triple=aarch64-pc-windows-msvc -filetype=obj -o /dev/null

declare void @g()
declare i32 @__CxxFrameHandler3(...)

define void @f() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %cont unwind label %catch.dispatch

catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %cont

cont:
  ret void
}

; CHECK:      adrp x0, .LBB0_1
; CHECK-NEXT: add x0, x0, :lo12:.LBB0_1
