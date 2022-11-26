; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a local, optimized into a GEP (e.g., &a - 12)
;  safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32, ptr %a, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr5) nounwind
  ret void
}

declare i32 @printf(ptr, ...)
