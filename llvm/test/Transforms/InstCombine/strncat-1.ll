; Test that the strncat libcall simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@null = constant [1 x i8] zeroinitializer
@null_hello = constant [7 x i8] c"\00hello\00"

declare ptr @strncat(ptr, ptr, i32)
declare i32 @puts(ptr)

define i32 @main() {
; CHECK-LABEL: @main(
; CHECK-NOT: call ptr @strncat
; CHECK: call i32 @puts

  %target = alloca [1024 x i8]
  store i8 0, ptr %target

  ; rslt1 = strncat(target, "hello\00")
  %rslt1 = call ptr @strncat(ptr %target, ptr @hello, i32 6)

  ; rslt2 = strncat(rslt1, "\00")
  %rslt2 = call ptr @strncat(ptr %rslt1, ptr @null, i32 42)

  ; rslt3 = strncat(rslt2, "\00hello\00")
  %rslt3 = call ptr @strncat(ptr %rslt2, ptr @null_hello, i32 42)

  call i32 @puts(ptr %rslt3)
  ret i32 0
}
