; Test that llvm-reduce can remove uninteresting operand bundles from calls.
;
; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=operand-bundles,attributes --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-ALL: declare i32 @f1(i32, i32)
declare i32 @f1(i32, i32)

; CHECK-FINAL-LABEL: define i32 @interesting(i32 %arg0, i32 %arg1) {
; CHECK-FINAL-NEXT:  entry:
; CHECK-FINAL-NEXT:    %r = call "attr0" i32 @f1(i32 "attr4" %arg0, i32 %arg1) #0
; CHECK-FINAL-NEXT:    ret i32 %r
; CHECK-FINAL-NEXT:  }
define i32 @interesting(i32 %arg0, i32 %arg1) {
entry:
; CHECK-INTERESTINGNESS-LABEL: @interesting(

; CHECK-INTERESTINGNESS: %r = call
; CHECK-INTERESTINGNESS-SAME: "attr0"
; CHECK-INTERESTINGNESS-SAME: i32 @f1(
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-INTERESTINGNESS-SAME: "attr4"
; CHECK-INTERESTINGNESS-SAME: %arg0
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-INTERESTINGNESS-SAME: %arg1
; CHECK-INTERESTINGNESS-SAME: #0
; CHECK-INTERESTINGNESS: ret i32 %r

  %r = call "attr0" "attr1" "attr2" i32 @f1(i32 "attr3" "attr4" "attr5" %arg0, i32 "attr6" "attr7" "attr8" %arg1) #0
  ret i32 %r
}

; CHECK-INTERESTINGNESS: attributes #0 = {
; CHECK-INTERESTINGNESS-SAME: "attr10"

; CHECK-FINAL:  attributes #0 = { "attr10" }

attributes #0 = { "attr9" "attr10" "attr11" }
