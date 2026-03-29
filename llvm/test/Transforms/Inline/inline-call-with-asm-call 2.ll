;; Test to verify that when callee has inline assembly, bumping up `-inline-asm-instr-cost` would block inlining.

; RUN: opt < %s -passes=inline -S | FileCheck %s --check-prefixes=CHECK,INLINE
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s --check-prefixes=CHECK,INLINE

;; Verify that a low assembly instruction cost of 150 does not block inlining.
;; This test also verifies that the outlined section's instructions (in "other"
;; section) do not contribute to the cost.
; RUN: opt < %s -passes=inline -inline-asm-instr-cost=150 -S | FileCheck %s --check-prefixes=CHECK,INLINE
; RUN: opt < %s -passes='cgscc(inline)' -inline-asm-instr-cost=150 -S | FileCheck %s --check-prefixes=CHECK,INLINE

;; Verify that an assembly instruction cost of 300 blocks inlining.
; RUN: opt < %s -passes=inline -inline-asm-instr-cost=300 -S | FileCheck %s --check-prefixes=CHECK,NOINLINE
; RUN: opt < %s -passes='cgscc(inline)' -inline-asm-instr-cost=300 -S | FileCheck %s --check-prefixes=CHECK,NOINLINE

define void @caller(i32 %a, i1 %b) #0 {
  call void @callee(i32 %a, i1 %b)
  ret void
}

; CHECK: define void @caller
; INLINE: call void asm
; NOINLINE: call void @callee


;; callee function with asm call with two real assembly instructions in the
;; destination section and two assembly instructions in the outlined "other"
;; section.
define void @callee(i32 %a, i1 %b) {
  call void asm sideeffect "s_nop 1\0A\09.pushsection other\0A\09s_nop 2\0A\09s_nop 3\0A\09.popsection\0A\09s_nop 4\0A\09.align 32", ""()
  ret void
}
; CHECK: define void @callee


