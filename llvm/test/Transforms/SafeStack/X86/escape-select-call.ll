; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of in select instruction
; safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, ptr %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, ptr %x, ptr null
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.1)
  ret void
}

declare double @testi_aux()
declare i32 @printf(ptr, ...)
