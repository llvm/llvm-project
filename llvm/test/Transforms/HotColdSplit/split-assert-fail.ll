; REQUIRES: asserts
; RUN: opt -S -instsimplify -hotcoldsplit -debug < %s 2>&1 | FileCheck %s
; RUN: opt -instcombine -hotcoldsplit -instsimplify %s -o /dev/null

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [2 x i8] c"0\00", align 1
@.str.1 = private unnamed_addr constant [14 x i8] c"assert-fail.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [15 x i8] c"int main(void)\00", align 1

; CHECK: @f
; CHECK-LABEL: codeRepl:
; CHECK }
; CHECK: define {{.*}}@f.cold.1()
; CHECK-LABEL: newFuncRoot:
; CHECK:   br label %if.then

; Function Attrs: nounwind willreturn
define i32 @f() #0 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %i, align 4
  %0 = load i32, i32* %i, align 4
  %cmp = icmp eq i32 %0, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @__assert_fail(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.1, i64 0, i64 0), i32 10, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #1
  unreachable

if.end:                                           ; preds = %entry
  %1 = load i32, i32* %i, align 4
  %add = add nsw i32 %1, 1
  store i32 %add, i32* %i, align 4
  %2 = load i32, i32* %i, align 4
  ret i32 %2
}

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) #1

attributes #0 = { nounwind willreturn }
attributes #1 = { noreturn nounwind }

