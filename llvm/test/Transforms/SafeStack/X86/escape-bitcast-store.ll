; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a local cast to a ptr of a different type
;   (e.g., int a; ... ; ptr b = &a;)
;  safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %a = alloca i32, align 4
  %b = alloca ptr, align 8
  store i32 0, ptr %a, align 4
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

declare i32 @printf(ptr, ...)
