; Test that llvm-reduce can remove uninteresting instructions.
;
; RUN: llvm-reduce --test %python --test-arg %p/Inputs/remove-instructions.py %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting %s
; REQUIRES: plugins

; We're testing all direct uses of %interesting are conserved
; CHECK-COUNT-5: %interesting
define i32 @main() #0 {
entry:
  %uninteresting1 = alloca i32, align 4
  %interesting = alloca i32, align 4
  %uninteresting2 = alloca i32, align 4
  store i32 0, i32* %uninteresting1, align 4
  store i32 0, i32* %interesting, align 4
  %0 = load i32, i32* %interesting, align 4
  %uninteresting3 = add nsw i32 %0, 1
  store i32 %uninteresting3, i32* %interesting, align 4
  %1 = load i32, i32* %interesting, align 4
  store i32 %1, i32* %uninteresting2, align 4
  ; CHECK-NOT: ret
  ret i32 0
}
