; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
;
; RUN: rm -rf %t
; RUN: mkdir %t
; get the python path from lit
; RUN: echo "#!" %python > %t/test.py
; then include the rest of the test script
; RUN: cat %p/Inputs/remove-args.py >> %t/test.py
; RUN: chmod +x %t/test.py

; RUN: llvm-reduce --test %t/test.py %s -o %t/out.ll
; RUN: cat %t/out.ll | FileCheck -implicit-check-not=uninteresting %s
; REQUIRES: shell

; CHECK: @interesting(i32 %interesting)
define void @interesting(i32 %uninteresting1, i32 %interesting, i32 %uninteresting2) {
entry:
  ; CHECK: call void @interesting(i32 0)
  call void @interesting(i32 -1, i32 0, i32 -1)
  ret void
}
