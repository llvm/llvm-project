; RUN: llc -mtriple=x86_64-pc-windows-msvc -stop-after=prolog-epilog -o - %s | FileCheck %s

; Check that "stack-protector-layout-only" is ignored for functions with
; funclets, which StackProtector does not support.

declare void @may_throw()
declare void @use(ptr)
declare i32 @__CxxFrameHandler3(...)

; CHECK-LABEL: name: funclet
; CHECK:       stackProtector:  ''
; CHECK-NEXT:  stackProtectorLayoutOnly: false
define void @funclet() sspstrong "stack-protector-layout-only" personality ptr @__CxxFrameHandler3 {
entry:
  %buf = alloca [16 x i8], align 1
  call void @use(ptr %buf)
  invoke void @may_throw()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %p to label %cont
}

; Control: the same function without funclets.
; CHECK-LABEL: name: no_funclet
; CHECK:       stackProtector:  ''
; CHECK-NEXT:  stackProtectorLayoutOnly: true
define void @no_funclet() sspstrong "stack-protector-layout-only" {
entry:
  %buf = alloca [16 x i8], align 1
  call void @use(ptr %buf)
  ret void
}
