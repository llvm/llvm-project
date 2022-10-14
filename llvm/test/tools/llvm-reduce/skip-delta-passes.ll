; RUN: llvm-reduce --delta-passes=attributes --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefix=RESULT %s < %t

; RUN: llvm-reduce --delta-passes=instructions,attributes --skip-delta-passes=instructions --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefix=RESULT %s < %t

; RUN: not llvm-reduce --skip-delta-passes=foo --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t 2>&1 | FileCheck %s --check-prefix=ERROR


; CHECK-INTERESTINGNESS: @foo
; RESULT: define void @foo() {
; RESULT-NEXT: store i32
; RESULT-NEXT: ret void
; RESULT0-NOT: attributes

; ERROR: unknown pass "foo"
define void @foo() #0 {
  store i32 0, ptr null
  ret void
}

attributes #0 = { "arstarstarst" }
