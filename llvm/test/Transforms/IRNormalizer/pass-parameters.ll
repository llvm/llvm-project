; RUN: not opt -S -passes='normalize<invalid>' %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: opt -S -passes=normalize < %s | FileCheck %s
; RUN: opt -S -passes='normalize<>' < %s | FileCheck %s
; RUN: opt -S -passes='normalize<preserve-order;rename-all;fold-all;reorder-operands>' < %s | FileCheck %s
; RUN: opt -S -passes='normalize<no-preserve-order;no-rename-all;no-fold-all;no-reorder-operands>' < %s | FileCheck %s

; FIXME: This verifies all the pass parameter names parse, but not
; that they work as expected.

; ERR: invalid normalize pass parameter 'invalid'

; CHECK: define i32 @0(i32 %a0, i32 %a1) {
; CHECK-NEXT: bb17254:
; CHECK-NEXT: %"vl12603(%a0, %a1)" = add i32 %a0, %a1
; CHECK-NEXT: ret i32 %"vl12603(%a0, %a1)"
; CHECK-NEXT: }
define i32 @0(i32, i32) {
  %3 = add i32 %0, %1
  ret i32 %3
}

