; RUN: not --crash llc -mtriple=x86_64-pc-windows-msvc < %s 2>&1 | FileCheck %s

; CHECK: Can't handle guaranteed tail calls that change the stack argument size in a function with EH funclets under win64 yet

declare i32 @__CxxFrameHandler3(...)

declare void @may_throw()

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define tailcc void @f(i64 %a, i64 %b, i1 %c) personality ptr @__CxxFrameHandler3 {
entry:
  br i1 %c, label %tail, label %try

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
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}
