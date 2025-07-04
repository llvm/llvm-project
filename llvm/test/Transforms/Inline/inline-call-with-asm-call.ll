;; Test to verify that when callee has inline assembly, bumping up `-inline-asm-instr-cost` would block inlining.

; RUN: opt < %s -passes=inline -S | FileCheck %s --check-prefixes=INLINE
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s --check-prefixes=INLINE
; RUN: opt < %s -passes=inline -inline-asm-instr-cost=150 -S | FileCheck %s --check-prefixes=INLINE
; RUN: opt < %s -passes='cgscc(inline)' -inline-asm-instr-cost=150 -S | FileCheck %s --check-prefixes=INLINE
; RUN: opt < %s -passes=inline -inline-asm-instr-cost=300 -S | FileCheck %s --check-prefixes=NOINLINE
; RUN: opt < %s -passes='cgscc(inline)' -inline-asm-instr-cost=300 -S | FileCheck %s --check-prefixes=NOINLINE

; CHECK-LABEL: caller
; CHECK-NOT: callee
; INLINE: call void asm
; NOINLINE: call void @callee

define void @caller(i32 %a, i1 %b) #0 {
  call void @callee(i32 %a, i1 %b)
  ret void
}

define void @callee(i32 %a, i1 %b) {
  call void asm sideeffect "s_nop 1\0A\09.pushsection other\0A\09s_nop 2\0A\09s_nop 3\0A\09.popsection\0A\09s_nop 4\0A\09.align 32", ""()
  ret void
}
