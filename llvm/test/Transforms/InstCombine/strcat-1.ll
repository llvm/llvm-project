; Test that the strcat libcall simplifier works correctly per the
; bug found in PR3661.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@null = constant [1 x i8] zeroinitializer
@null_hello = constant [7 x i8] c"\00hello\00"

declare ptr @strcat(ptr, ptr)
declare i32 @puts(ptr)

define i32 @main() {
; CHECK-LABEL: @main(
; CHECK-NOT: call ptr @strcat
; CHECK: call i32 @puts

  %target = alloca [1024 x i8]
  store i8 0, ptr %target

  ; rslt1 = strcat(target, "hello\00")
  %rslt1 = call ptr @strcat(ptr %target, ptr @hello)

  ; rslt2 = strcat(rslt1, "\00")
  %rslt2 = call ptr @strcat(ptr %rslt1, ptr @null)

  ; rslt3 = strcat(rslt2, "\00hello\00")
  %rslt3 = call ptr @strcat(ptr %rslt2, ptr @null_hello)

  call i32 @puts( ptr %rslt3 )
  ret i32 0
}
