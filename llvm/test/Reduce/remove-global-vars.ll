; Test that llvm-reduce can remove uninteresting Global Variables as well as
; their direct uses (which in turn are replaced with 'undef').
;
; RUN: rm -rf %t
; RUN: mkdir %t
; get the python path from lit
; RUN: echo "#!" %python > %t/test.py
; then include the rest of the test script
; RUN: cat %p/Inputs/remove-global-vars.py >> %t/test.py
; RUN: chmod +x %t/test.py

; RUN: llvm-reduce --test %t/test.py %s -o %t/out.ll
; RUN: cat %t/out.ll | FileCheck -implicit-check-not=uninteresting %s
; REQUIRES: shell

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

  ; CHECK: %inc = add nsw i32 undef, 1
  %inc = add nsw i32 %0, 1
  ; CHECK: store i32 %inc, i32* @interesting, align 4
  store i32 %inc, i32* @interesting, align 4
  ret i32 0
}
