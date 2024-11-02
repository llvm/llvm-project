; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s --check-prefix CHECK
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s --check-prefix CHECK

@i1 = external constant i32 #0
@i2 = constant ptr @i1 #0

define i32 @read() {
  %1  = load i32, ptr @i1, align 4
  ret i32 %1
}

define ptr @retptr() {
  ret ptr @i2
}

; CHECK:       .read:
; CHECK:        la 3, i1[TD](2)

; CHECK:       .retptr:
; CHECK:        la 3, i2[TD](2)

; CHECK-DAG:   .toc
; CHECK:         .extern i1[TD]
; CHECK:         .csect i2[TD]

attributes #0 = { "toc-data" }
