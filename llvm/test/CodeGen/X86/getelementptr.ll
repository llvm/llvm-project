; RUN: llc < %s -O0 -mtriple=i686--
; RUN: llc < %s -O0 -mtriple=x86_64--
; RUN: llc < %s -O2 -mtriple=i686--
; RUN: llc < %s -O2 -mtriple=x86_64--


; Test big index trunc to pointer size:

define ptr @test_trunc65(ptr %ptr) nounwind {
; CHECK-LABEL: test_trunc65
; CHECK: 3
  %d = getelementptr i8, ptr %ptr, i65 18446744073709551619 ; 2^64 + 3
  ret ptr %d
}

define ptr @test_trunc128(ptr %ptr) nounwind {
; CHECK-LABEL: test_trunc128
; CHECK: 5
  %d = getelementptr i8, ptr %ptr, i128 18446744073709551621 ; 2^64 + 5
  ret ptr %d
}

define ptr @test_trunc160(ptr %ptr) nounwind {
; CHECK-LABEL: test_trunc160
; CHECK: 8
  %d = getelementptr i8, ptr %ptr, i160 18446744073709551624 ; 2^64 + 8
  ret ptr %d
}

define ptr @test_trunc256(ptr %ptr) nounwind {
; CHECK-LABEL: test_trunc256
; CHECK: 13
  %d = getelementptr i8, ptr %ptr, i256 18446744073709551629 ; 2^64 + 13
  ret ptr %d
}

define ptr @test_trunc2048(ptr %ptr) nounwind {
; CHECK-LABEL: test_trunc2048
; CHECK: 21
  %d = getelementptr i8, ptr %ptr, i2048 18446744073709551637 ; 2^64 + 21
  ret ptr %d
}


; Test small index sext to pointer size

define ptr @test_sext3(ptr %ptr) nounwind {
; CHECK-LABEL: test_sext3
; CHECK: -3
  %d = getelementptr i8, ptr %ptr, i3 -3
  ret ptr %d
}

define ptr @test_sext5(ptr %ptr) nounwind {
; CHECK-LABEL: test_sext5
; CHECK: -5
  %d = getelementptr i8, ptr %ptr, i5 -5
  ret ptr %d
}

define ptr @test_sext8(ptr %ptr) nounwind {
; CHECK-LABEL: test_sext8
; CHECK: -8
  %d = getelementptr i8, ptr %ptr, i8 -8
  ret ptr %d
}

define ptr @test_sext13(ptr %ptr) nounwind {
; CHECK-LABEL: test_sext13
; CHECK: -13
  %d = getelementptr i8, ptr %ptr, i8 -13
  ret ptr %d
}

define ptr @test_sext16(ptr %ptr) nounwind {
; CHECK-LABEL: test_sext16
; CHECK: -21
  %d = getelementptr i8, ptr %ptr, i8 -21
  ret ptr %d
}


; Test out of int64_t range indices

; OSS-Fuzz: https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=7173
define void @test_outofrange(ptr %ptr) nounwind {
; CHECK-LABEL: test_outofrange
  %d = getelementptr i96, ptr %ptr, i96 39614081257132168796771975167
  %ld = load i96, ptr %d, align 1
  unreachable
}
