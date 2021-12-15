; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s

define i32 @f_1(i32 %x) {
; CHECK-LABEL: @f_1(
wentry:
  %cond = icmp ugt i32 %x, 0
  br i1 %cond, label %return, label %body

body:
; CHECK: body:
; CHECK: call i32 @f_1(i32 %y) [ "deopt"() ]
  %y = add i32 %x, 1
  %tmp = call i32 @f_1(i32 %y) [ "deopt"() ]
  ret i32 0

return:
  ret i32 1
}

define i32 @f_2(i32 %x) {
; CHECK-LABEL: @f_2

entry:
  %cond = icmp ugt i32 %x, 0
  br i1 %cond, label %return, label %body

body:
; CHECK: body:
; CHECK: call i32 @f_2(i32 %y) [ "unknown"() ]
  %y = add i32 %x, 1
  %tmp = call i32 @f_2(i32 %y) [ "unknown"() ]
  ret i32 0

return:
  ret i32 1
}

declare void @func()

define void @f_3(i1 %B) personality i8 42 {
; CHECK-LABEL: @f_3(
entry:
  invoke void @func()
          to label %exit unwind label %merge
merge:
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:
; CHECK: catch:
; CHECK: call void @f_3(i1 %B) [ "funclet"(token %cp) ]
  %cp = catchpad within %cs1 []
  call void @f_3(i1 %B) [ "funclet"(token %cp) ]
  ret void

exit:
  ret void
}

; CHECK-LABEL: @test_clang_arc_attachedcall(
; CHECK: tail call i8* @getObj(

declare i8* @getObj()

define i8* @test_clang_arc_attachedcall() {
  %r = call i8* @getObj() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret i8* %r
}

declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
