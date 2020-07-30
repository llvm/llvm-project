; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
;
; RUN: llvm-reduce --test %python --test-arg %p/Inputs/remove-multiple-use-of-global-vars-in-same-instruction.py %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting %s

; CHECK: @uninteresting1 = global
; CHECK: @uninteresting2 = global
; CHECK: @uninteresting3 = global
@uninteresting1 = global i32 0, align 4
@uninteresting2 = global i32 0, align 4
@uninteresting3 = global i32 0, align 4

declare void @use(i32*, i32*, i32*)

; CHECK-LABEL: @interesting()
define void @interesting() {
entry:
  ; CHECK: call void @use(i32* @uninteresting1, i32* @uninteresting2, i32* @uninteresting3)
  call void @use(i32* @uninteresting1, i32* @uninteresting2, i32* @uninteresting3)
  call void @use(i32* @uninteresting1, i32* @uninteresting2, i32* @uninteresting3)
  call void @use(i32* @uninteresting1, i32* @uninteresting2, i32* @uninteresting3)
  ret void
}
