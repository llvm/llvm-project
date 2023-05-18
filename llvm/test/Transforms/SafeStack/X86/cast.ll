; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; PtrToInt/IntToPtr Cast

define void @IntToPtr() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @IntToPtr(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a = alloca i32, align 4
  %0 = ptrtoint ptr %a to i64
  %1 = inttoptr i64 %0 to ptr
  ret void
}

define i8 @BitCastNarrow() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @BitCastNarrow(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %a = alloca i32, align 4
  %0 = load i8, ptr %a, align 1
  ret i8 %0
}

define i64 @BitCastWide() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @BitCastWide(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i64
  %a = alloca i32, align 4
  %0 = load i64, ptr %a, align 1
  ret i64 %0
}
