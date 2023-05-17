; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%struct.pair = type { i32, i32 }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of struct element. (GEP followed by store).
;  safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  store ptr %y, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

declare i32 @printf(ptr, ...)
