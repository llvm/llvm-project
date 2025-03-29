; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; array of [16 x i8]

define void @foo(ptr %a) nounwind uwtable safestack {
entry:
  ; CHECK: %[[USP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr

  ; CHECK: %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16

  ; CHECK: store ptr %[[USST]], ptr @__safestack_unsafe_stack_ptr

  %a.addr = alloca ptr, align 8
  %buf = alloca [16 x i8], align 16

  ; CHECK: %[[AADDR:.*]] = alloca ptr, align 8
  ; CHECK: store ptr {{.*}}, ptr %[[AADDR]], align 8
  store ptr %a, ptr %a.addr, align 8

  ; CHECK: %[[A2:.*]] = load ptr, ptr %[[AADDR]], align 8
  ; CHECK: %[[BUFPTR:.*]] = getelementptr i8, ptr %[[USP]], i32 -16
  %a2 = load ptr, ptr %a.addr, align 8

  ; CHECK: call ptr @strcpy(ptr %[[BUFPTR]], ptr %[[A2]])
  %call = call ptr @strcpy(ptr %buf, ptr %a2)

  ; CHECK: store ptr %[[USP]], ptr @__safestack_unsafe_stack_ptr
  ret void
}

declare ptr @strcpy(ptr, ptr)
