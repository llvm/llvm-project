; RUN: llc -mtriple powerpc64-unknown-linux-gnu -fast-isel -O0 < %s | FileCheck %s

; Verify that pointer offsets larger than 32 bits work correctly.

define void @test(i32* %array) {
; CHECK-LABEL: test:
; CHECK-NOT: li {{[0-9]+}}, -8
  %element = getelementptr i32, i32* %array, i64 2147483646
  store i32 1234, i32* %element
  ret void
}

