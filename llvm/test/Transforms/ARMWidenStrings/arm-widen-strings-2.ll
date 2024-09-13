; RUN: opt < %s -mtriple=arm-arm-none-eabi -passes="default<O2>,globalopt" -S | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK: [64 x i8]
@.str = private unnamed_addr constant [62 x i8] c"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\00", align 1

; Function Attrs: nounwind
define hidden void @foo() #0 {
entry:
; CHECK: %something = alloca [64 x i8]
  %something = alloca [62 x i8], align 1
  %arraydecay = getelementptr inbounds [62 x i8], ptr %something, i32 0, i32 0
; CHECK: @llvm.memcpy.p0.p0.i32
  %call = call ptr @strcpy(ptr %arraydecay, ptr @.str)
  %arraydecay1 = getelementptr inbounds [62 x i8], ptr %something, i32 0, i32 0
  %call2 = call i32 @bar(ptr %arraydecay1)
  ret void
}

declare ptr @strcpy(ptr, ptr) #1

declare i32 @bar(...) #1
