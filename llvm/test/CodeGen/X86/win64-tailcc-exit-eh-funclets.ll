; RUN: not --crash llc -mtriple=x86_64-pc-windows-msvc %s -o - 2>&1 | FileCheck %s

; A tail call that only shrinks the stack argument area uses the same exit as
; one with a reserve, which is not handled in functions with EH funclets.

; CHECK: Can't handle guaranteed tail calls that change the stack argument size in a function with EH funclets under win64 yet

declare i32 @__CxxFrameHandler3(...)

declare void @may_throw()

declare tailcc void @h(i64, i64, i64, i64, i64, i64, i64)

define tailcc void @f(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i1 %cc) personality ptr @__CxxFrameHandler3 {
entry:
  br i1 %cc, label %tail, label %try

try:
  invoke void @may_throw() to label %done unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %handler] unwind to caller

handler:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %done

done:
  ret void

tail:
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g)
  ret void
}
