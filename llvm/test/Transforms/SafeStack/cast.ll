; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; PtrToInt/IntToPtr Cast
; Requires no protector.

; CHECK-LABEL: @foo(
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  %a = alloca i32, align 4
  %0 = ptrtoint i32* %a to i64
  %1 = inttoptr i64 %0 to i32*
  ret void
}
