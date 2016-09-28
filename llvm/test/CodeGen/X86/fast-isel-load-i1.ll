; RUN: llc < %s -fast-isel -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f  | FileCheck %s

define i1 @test_i1(i1* %b) {
; CHECK-LABEL: test_i1:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    testb $1, (%rdi)
entry:
  %0 = load i1, i1* %b, align 1
  br i1 %0, label %in, label %out
in:
  ret i1 0
out:
  ret i1 1
}

