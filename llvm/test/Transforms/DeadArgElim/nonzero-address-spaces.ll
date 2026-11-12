; RUN: opt -S -passes=deadargelim %s | FileCheck %s

; DeadArgumentElimination should respect the function address space
; in the data layout.

target datalayout = "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8"

; CHECK: define internal i32 @foo() addrspace(1)
define internal i32 @foo(i32 %x) #0 {
  tail call void asm sideeffect inteldialect "mov eax, [esp + $$4]\0A\09ret", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  unreachable
}

define i32 @f(i32 %x, i32 %y) {
  ; CHECK: %r = call addrspace(1) i32 @foo()
  %r = call i32 @foo(i32 %x)

  ret i32 %r
}

