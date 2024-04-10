; Test that llvm-reduce can remove uninteresting attributes.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-ALL: @gv0 = global i32 0 #0
; CHECK-ALL-NEXT: @gv1 = global i32 0 #1
; CHECK-ALL-NEXT: @gv2 = global i32 0
@gv0 = global i32 0 #0
@gv1 = global i32 0 #1
@gv2 = global i32 0 #2

; CHECK-INTERESTINGNESS: attributes #0 = {
; CHECK-INTERESTINGNESS-SAME: "attr0"
; CHECK-INTERESTINGNESS-SAME: "attr2"

; CHECK-INTERESTINGNESS-NEXT: attributes #1 = {
; CHECK-INTERESTINGNESS-SAME: "attr4"

; CHECK-FINAL:  attributes #0 = { "attr0" "attr2" }
; CHECK-FINAL-NEXT:  attributes #1 = { "attr4" }

; CHECK-FINAL-NOT:  attributes #2

attributes #0 = { "attr0" "attr1" "attr2"}
attributes #1 = { "attr3" "attr4" "attr5"}
attributes #2 = { "attr6" "attr7" "attr8"}
