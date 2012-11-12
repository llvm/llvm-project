; RUN: not llc --help-hidden 2>&1 | FileCheck %s

; CHECK: -print-after
; CHECK-NOT: -print-after-all
; CHECK: =simple-register-coalescing
; CHECK: -print-after-all
