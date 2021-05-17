; Test that llvm-reduce can remove uninteresting functions as well as
; their InstCalls.
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting --check-prefixes=CHECK-ALL,CHECK-FINAL %s

define i32 @uninteresting1() {
entry:
  ret i32 0
}

; CHECK-ALL-LABEL: interesting()
define i32 @interesting() {
entry:
  ; CHECK-INTERESTINGNESS: call i32 @interesting()
  %call2 = call i32 @interesting()
  %call = call i32 @uninteresting1()
  ret i32 5
}

; CHECK-FINAL-NEXT: entry:
; CHECK-FINAL-NEXT:   %call2 = call i32 @interesting()
; CHECK-FINAL-NEXT:   ret i32 5
; CHECK-FINAL-NEXT: }

define i32 @uninteresting2() {
entry:
  ret i32 0
}

declare void @uninteresting3()
