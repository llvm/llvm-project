; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=+sse2 | FileCheck %s

define void @test() {
; CHECK-LABEL: test:
; CHECK:       # BB#0:
; CHECK-NEXT:    mfence
; CHECK-NEXT:    retl
  fence seq_cst
  ret void
}

