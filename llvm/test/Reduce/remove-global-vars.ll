; Test that llvm-reduce can remove uninteresting Global Variables as well as
; their direct uses (which in turn are replaced with 'undef').
;
; RUN: llvm-reduce --test %python --test-arg %p/Inputs/remove-global-vars.py %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting %s

; CHECK: @interesting = global
@interesting = global i32 0, align 4
@uninteresting = global i32 1, align 4

define i32 @main() {
entry:
  %0 = load i32, i32* @uninteresting, align 4
  ; CHECK: store i32 undef, i32* @interesting, align 4
  store i32 %0, i32* @interesting, align 4

  ; CHECK: load i32, i32* @interesting, align 4
  %1 = load i32, i32* @interesting, align 4
  store i32 %1, i32* @uninteresting, align 4

  ; CHECK: store i32 5, i32* @interesting, align 4
  store i32 5, i32* @interesting, align 4
  ret i32 0
}
