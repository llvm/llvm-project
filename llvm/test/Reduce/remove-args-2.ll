; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
; This test checks that functions with different argument types are handled correctly
;
; RUN: llvm-reduce --test %python --test-arg %p/Inputs/remove-args.py %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting %s

%struct.foo = type { %struct.foo*, i32, i32, i8* }

define dso_local void @bar() {
entry:
  ; CHECK: call void @interesting(%struct.foo* null)
  call void @interesting(i32 0, i8* null, %struct.foo* null, i8* null, i64 0)
  ret void
}

; CHECK: define internal void @interesting(%struct.foo* %interesting) {
define internal void @interesting(i32 %uninteresting1, i8* %uninteresting2, %struct.foo* %interesting, i8* %uninteresting3, i64 %uninteresting4) {
entry:
  ret void
}
