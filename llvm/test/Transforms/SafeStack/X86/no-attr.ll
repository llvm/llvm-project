; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; no safestack attribute
; Requires no protector.

; CHECK-NOT: __safestack_unsafe_stack_ptr

; CHECK: @foo
define void @foo(ptr %a) nounwind uwtable {
entry:
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  %a.addr = alloca ptr, align 8
  %buf = alloca [16 x i8], align 16
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

declare ptr @strcpy(ptr, ptr)
declare i32 @printf(ptr, ...)
