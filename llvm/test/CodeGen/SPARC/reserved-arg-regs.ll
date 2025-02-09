;; Test reserving argument registers.
; RUN: not llc < %s -mtriple=sparc-linux-gnu -mattr=+reserve-o0 2>&1 | FileCheck %s --check-prefixes=CHECK-RESERVED-O0
; RUN: not llc < %s -mtriple=sparc64-linux-gnu -mattr=+reserve-o0 2>&1 | FileCheck %s --check-prefixes=CHECK-RESERVED-O0
; RUN: not llc < %s -mtriple=sparc-linux-gnu -mattr=+reserve-i0 2>&1 | FileCheck %s --check-prefixes=CHECK-RESERVED-I0
; RUN: not llc < %s -mtriple=sparc64-linux-gnu -mattr=+reserve-i0 2>&1 | FileCheck %s --check-prefixes=CHECK-RESERVED-I0

; CHECK-RESERVED-O0: error:
; CHECK-RESERVED-O0-SAME: SPARC doesn't support function calls if any of the argument registers is reserved.
; CHECK-RESERVED-I0: error:
; CHECK-RESERVED-I0-SAME: SPARC doesn't support function calls if any of the argument registers is reserved.
define void @call_function() {
  call void @foo()
  ret void
}
declare void @foo()

; CHECK-RESERVED-O0: error:
; CHECK-RESERVED-O0-SAME: SPARC doesn't support function calls if any of the argument registers is reserved.
; CHECK-RESERVED-I0: error:
; CHECK-RESERVED-I0-SAME: SPARC doesn't support function calls if any of the argument registers is reserved.
define void @call_function_with_arg(i8 %in) {
  call void @bar(i8 %in)
  ret void
}
declare void @bar(i8)
