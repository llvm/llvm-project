; RUN: llc -mtriple=x86_64 < %s | FileCheck %s

define i1 @query() {
; CHECK-LABEL: query:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %enabled = call i1 @llvm.is.debugging.enabled()
  ret i1 %enabled
}

declare i1 @llvm.is.debugging.enabled()
