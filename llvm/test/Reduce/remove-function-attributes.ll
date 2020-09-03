; Test that llvm-reduce can remove uninteresting attributes.
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-INTERESTINGNESS: declare
; CHECK-INTERESTINGNESS-SAME: "attr0"
; CHECK-INTERESTINGNESS-SAME: void @f0
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-INTERESTINGNESS-SAME: "attr6"
; CHECK-INTERESTINGNESS-SAME: #0

; CHECK-FINAL: declare "attr0" void @f0(i32, i32 "attr6") #0

declare "attr0" "attr1" "attr2" void @f0(i32 "attr3" "attr4" "attr5", i32 "attr6" "attr7" "attr8") #0

; CHECK-INTERESTINGNESS: attributes #0 = {
; CHECK-INTERESTINGNESS-SAME: "attr10"

; CHECK-FINAL:  attributes #0 = { "attr10" }

attributes #0 = { "attr9" "attr10" "attr11" }
