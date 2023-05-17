; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a casted pointer
;  safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funfloat2(ptr %0)
  ret void
}

declare void @funfloat2(ptr)
declare ptr @getp()
