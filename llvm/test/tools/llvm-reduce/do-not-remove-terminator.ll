; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-FINAL %s

; Make sure we do not remove the terminator of the entry block. The interesting
; check only requires the result to define the function @test.

; Test case for PR43798.

; CHECK-INTERESTINGNESS: define i32 @test

; CHECK-FINAL:      define i32 @test
; CHECK-FINAL-NEXT: entry:
; CHECK-FINAL-NEXT:   ret i32

define i32 @test(i32 %x) {
entry:
  %add = add i32 %x, %x
  ret i32 %add
}
