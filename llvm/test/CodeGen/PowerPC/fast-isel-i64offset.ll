; RUN: llc -verify-machineinstrs -mtriple powerpc64-unknown-linux-gnu -fast-isel -O0 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -fast-isel -O0 < %s | FileCheck %s

; Verify that pointer offsets larger than 32 bits work correctly.

define void @test(ptr %array) {
; CHECK-LABEL: test:
; CHECK-NOT: li {{[0-9]+}}, -8
  %element = getelementptr i32, ptr %array, i64 2147483646
  store i32 1234, ptr %element
  ret void
}

