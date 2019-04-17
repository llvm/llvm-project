; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; PtrToInt/IntToPtr Cast

define void @IntToPtr() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @IntToPtr(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a = alloca i32, align 4
  %0 = ptrtoint i32* %a to i64
  %1 = inttoptr i64 %0 to i32*
  ret void
}

define i8 @BitCastNarrow() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @BitCastNarrow(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %a = alloca i32, align 4
  %0 = bitcast i32* %a to i8*
  %1 = load i8, i8* %0, align 1
  ret i8 %1
}

define i64 @BitCastWide() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @BitCastWide(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i64
  %a = alloca i32, align 4
  %0 = bitcast i32* %a to i64*
  %1 = load i64, i64* %0, align 1
  ret i64 %1
}
