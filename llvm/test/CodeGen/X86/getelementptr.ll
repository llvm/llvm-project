; RUN: llc < %s -O0 -mtriple=i686--
; RUN: llc < %s -O0 -mtriple=x86_64--
; RUN: llc < %s -O2 -mtriple=i686--
; RUN: llc < %s -O2 -mtriple=x86_64--


; Test big index trunc to pointer size:

define i8* @test_trunc65(i8* %ptr) nounwind {
; CHECK-LABEL: test_trunc65
; CHECK: 3
  %d = getelementptr i8, i8* %ptr, i65 18446744073709551619 ; 2^64 + 3
  ret i8* %d
}

define i8* @test_trunc128(i8* %ptr) nounwind {
; CHECK-LABEL: test_trunc128
; CHECK: 5
  %d = getelementptr i8, i8* %ptr, i128 18446744073709551621 ; 2^64 + 5
  ret i8* %d
}

define i8* @test_trunc160(i8* %ptr) nounwind {
; CHECK-LABEL: test_trunc160
; CHECK: 8
  %d = getelementptr i8, i8* %ptr, i160 18446744073709551624 ; 2^64 + 8
  ret i8* %d
}

define i8* @test_trunc256(i8* %ptr) nounwind {
; CHECK-LABEL: test_trunc256
; CHECK: 13
  %d = getelementptr i8, i8* %ptr, i256 18446744073709551629 ; 2^64 + 13
  ret i8* %d
}

define i8* @test_trunc2048(i8* %ptr) nounwind {
; CHECK-LABEL: test_trunc2048
; CHECK: 21
  %d = getelementptr i8, i8* %ptr, i2048 18446744073709551637 ; 2^64 + 21
  ret i8* %d
}


; Test small index sext to pointer size

define i8* @test_sext3(i8* %ptr) nounwind {
; CHECK-LABEL: test_sext3
; CHECK: -3
  %d = getelementptr i8, i8* %ptr, i3 -3
  ret i8* %d
}

define i8* @test_sext5(i8* %ptr) nounwind {
; CHECK-LABEL: test_sext5
; CHECK: -5
  %d = getelementptr i8, i8* %ptr, i5 -5
  ret i8* %d
}

define i8* @test_sext8(i8* %ptr) nounwind {
; CHECK-LABEL: test_sext8
; CHECK: -8
  %d = getelementptr i8, i8* %ptr, i8 -8
  ret i8* %d
}

define i8* @test_sext13(i8* %ptr) nounwind {
; CHECK-LABEL: test_sext13
; CHECK: -13
  %d = getelementptr i8, i8* %ptr, i8 -13
  ret i8* %d
}

define i8* @test_sext16(i8* %ptr) nounwind {
; CHECK-LABEL: test_sext16
; CHECK: -21
  %d = getelementptr i8, i8* %ptr, i8 -21
  ret i8* %d
}
