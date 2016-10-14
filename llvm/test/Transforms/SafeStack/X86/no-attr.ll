; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; no safestack attribute
; Requires no protector.

; CHECK-NOT: __safestack_unsafe_stack_ptr

; CHECK: @foo
define void @foo(i8* %a) nounwind uwtable {
entry:
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8*, i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8], [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

declare i8* @strcpy(i8*, i8*)
declare i32 @printf(i8*, ...)
