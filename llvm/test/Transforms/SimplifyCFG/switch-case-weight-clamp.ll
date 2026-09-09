; RUN: split-file %s %t
; RUN: opt -S -passes=simplifycfg %t/count.ll | FileCheck %s --check-prefix=COUNT
; RUN: opt -S -passes=simplifycfg %t/expected.ll | FileCheck %s --check-prefix=EXPECTED
;
; Overflowing the i32 weight sum marks approxprofile. llvm.expect does not.

; COUNT: Function Attrs: approxprofile
; COUNT-LABEL: define i32 @count(
; COUNT: attributes #[[ATTR:[0-9]+]] = { approxprofile }

; EXPECTED-NOT: Function Attrs: approxprofile
; EXPECTED-NOT: approxprofile
; EXPECTED-LABEL: define i32 @expected(i32 %x) {

;--- count.ll
define i32 @count(i32 %x) !prof !0 {
entry:
  switch i32 %x, label %def [
    i32 42, label %def
    i32 0, label %a
  ], !prof !1
def:
  ret i32 0
a:
  ret i32 1
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 3000000000, i32 3000000000, i32 1}

;--- expected.ll
define i32 @expected(i32 %x) {
entry:
  switch i32 %x, label %def [
    i32 42, label %def
    i32 0, label %a
  ], !prof !0
def:
  ret i32 0
a:
  ret i32 1
}

!0 = !{!"branch_weights", !"expected", i32 3000000000, i32 3000000000, i32 1}
