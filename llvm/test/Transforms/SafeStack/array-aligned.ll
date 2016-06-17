; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; array of [16 x i8]

define void @foo(i8* %a) nounwind uwtable safestack {
entry:
  ; CHECK: %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr

  ; CHECK: %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16

  ; CHECK: store i8* %[[USST]], i8** @__safestack_unsafe_stack_ptr

  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16

  ; CHECK: %[[AADDR:.*]] = alloca i8*, align 8
  ; CHECK: store i8* {{.*}}, i8** %[[AADDR]], align 8
  store i8* %a, i8** %a.addr, align 8

  ; CHECK: %[[BUFPTR:.*]] = getelementptr i8, i8* %[[USP]], i32 -16
  ; CHECK: %[[BUFPTR2:.*]] = bitcast i8* %[[BUFPTR]] to [16 x i8]*
  ; CHECK: %[[GEP:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[BUFPTR2]], i32 0, i32 0
  %gep = getelementptr inbounds [16 x i8], [16 x i8]* %buf, i32 0, i32 0

  ; CHECK: %[[A2:.*]] = load i8*, i8** %[[AADDR]], align 8
  %a2 = load i8*, i8** %a.addr, align 8

  ; CHECK: call i8* @strcpy(i8* %[[GEP]], i8* %[[A2]])
  %call = call i8* @strcpy(i8* %gep, i8* %a2)

  ; CHECK: store i8* %[[USP]], i8** @__safestack_unsafe_stack_ptr
  ret void
}

declare i8* @strcpy(i8*, i8*)
