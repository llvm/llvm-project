; RUN: llvm-split -enable-call-graph-split-module=true -j10 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: not test -e %t2

; Test that partition count is capped to the number of entry functions
; (-j10 but only 2 roots → 2 outputs, no %t2 or beyond).

; CHECK0: define void @foo()
; CHECK1: define void @bar()

define void @foo() { ret void }
define void @bar() { ret void }
