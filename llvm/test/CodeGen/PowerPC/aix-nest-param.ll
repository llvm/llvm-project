; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

define ptr @nest_receiver(ptr nest %arg) nounwind {
  ret ptr %arg
}

define ptr @nest_caller(ptr %arg) nounwind {
  %result = call ptr @nest_receiver(ptr nest %arg)
  ret ptr %result
}
; CHECK-LABEL: .nest_receiver:
; CHECK:         mr      3, 11
; CHECK:         blr

; CHECK-LABEL: .nest_caller:
; CHECK:         mr      11, 3
; CHECK:         bl .nest_receiver
