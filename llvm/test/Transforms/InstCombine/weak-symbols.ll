; PR4738 - Test that the library call simplifier doesn't assume anything about
; weak symbols.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

@real_init = weak_odr constant [2 x i8] c"y\00"
@fake_init = weak constant [2 x i8] c"y\00"
@.str = private constant [2 x i8] c"y\00"

define i32 @foo() nounwind {
; CHECK-LABEL: define i32 @foo(
; CHECK: call i32 @strcmp
; CHECK: ret i32 %temp1

entry:
  %temp1 = call i32 @strcmp(ptr @fake_init, ptr @.str) nounwind readonly
  ret i32 %temp1
}

define i32 @bar() nounwind {
; CHECK-LABEL: define i32 @bar(
; CHECK: ret i32 0

entry:
  %temp1 = call i32 @strcmp(ptr @real_init, ptr @.str) nounwind readonly
  ret i32 %temp1
}

declare i32 @strcmp(ptr, ptr) nounwind readonly
