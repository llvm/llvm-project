; RUN: opt < %s -argpromotion -S | FileCheck %s

; ArgumentPromotion should preserve the default function address space
; from the data layout.

target datalayout = "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8"

@g = common global i32 0, align 4

define i32 @bar() {
entry:
  %call = call i32 @foo(i32* @g)
; CHECK: %call = call addrspace(1) i32 @foo()
  ret i32 %call
}

; CHECK: define internal i32 @foo() addrspace(1)
define internal i32 @foo(i32*) {
entry:
  %retval = alloca i32, align 4
  call void asm sideeffect "ldr r0, [r0] \0Abx lr        \0A", ""()
  unreachable
}

