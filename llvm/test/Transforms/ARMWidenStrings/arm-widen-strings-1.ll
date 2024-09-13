; RUN: opt < %s -mtriple=arm-arm-none-eabi -passes="default<O2>,globalopt" -S | FileCheck %s
; RUN: opt < %s -mtriple=arm-arm-none-eabi -passes="default<O0>" -S | FileCheck %s --check-prefix=TURNED-OFF
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK: [12 x i8]
; TURNED-OFF-NOT: [12 x i8]
@.str = private unnamed_addr constant [10 x i8] c"123456789\00", align 1

; Function Attrs: nounwind
define hidden void @foo() #0 {
entry:
; CHECK: %something = alloca [12 x i8]
; TURNED-OFF-NOT: %something = alloca [12 x i8]
  %something = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], ptr %something, i32 0, i32 0
; CHECK: @llvm.memcpy.p0.p0.i32
  %call = call ptr @strcpy(ptr %arraydecay, ptr @.str)
  %arraydecay1 = getelementptr inbounds [10 x i8], ptr %something, i32 0, i32 0
  %call2 = call i32 @bar(ptr %arraydecay1)
  ret void
}

declare ptr @strcpy(ptr, ptr) #1

declare i32 @bar(...) #1
