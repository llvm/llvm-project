; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
;
; RUN: llvm-reduce --test %python --test-arg %p/Inputs/remove-multiple-use-of-args-in-same-instruction.py %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting %s

declare void @use(i32, i32, i32)

; CHECK-LABEL: @interesting(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3
define void @interesting(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3) {
entry:
  ; CHECK: call void @use(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3)
  call void @use(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3)
  call void @use(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3)
  call void @use(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3)
  ret void
}
