; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=simplify-instructions --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=RESULT %s < %t

; CHECK-INTERESTINGNESS: ret

; RESULT: %add4 = add i32 %arg0, %arg1
; RESULT: ret i32 %add4

define i32 @func(i32 %arg0, i32 %arg1) {
entry:
  %add0 = add i32 %arg0, 0
  %add1 = add i32 %add0, 0
  %add2 = add i32 %add1, 0
  %add3 = add i32 %arg1, 0
  %add4 = add i32 %add2, %add3
  ret i32 %add4
}
