; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: declare

; CHECK-FINAL-NOT: module asm
; CHECK-FINAL: declare void @g

module asm "foo"

declare void @g()
