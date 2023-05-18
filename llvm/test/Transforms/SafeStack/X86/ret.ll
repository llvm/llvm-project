; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Returns an alloca address.
; Requires protector.

define i64 @foo() nounwind readnone safestack {
entry:
  ; CHECK-LABEL: define i64 @foo(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i64
  %x = alloca [100 x i32], align 16
  %0 = ptrtoint ptr %x to i64
  ret i64 %0
}
