; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
;
; RUN: llvm-reduce --abort-on-invalid-reduction -delta-passes=arguments --test %python --test-arg %p/Inputs/remove-args.py %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting %s

; CHECK: @interesting(i32 %interesting)
define void @interesting(i32 %uninteresting1, i32 %interesting, i32 %uninteresting2) {
entry:
  ; CHECK: call void @interesting(i32 0)
  call void @interesting(i32 -1, i32 0, i32 -1)
  ret void
}
