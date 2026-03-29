; RUN: opt -S -passes=lower-target-intrinsics -mtriple=x86_64-unknown-linux -mcpu=haswell < %s | FileCheck %s --check-prefix=HASWELL
; RUN: opt -S -passes=lower-target-intrinsics -mtriple=x86_64-unknown-linux -mcpu=x86-64 < %s | FileCheck %s --check-prefix=GENERIC

define i1 @test_has_avx2() {
; HASWELL-LABEL: @test_has_avx2(
; HASWELL-NEXT:    ret i1 true
;
; GENERIC-LABEL: @test_has_avx2(
; GENERIC-NEXT:    ret i1 false
  %1 = call i1 @llvm.target.has.feature(metadata !"avx2")
  ret i1 %1
}

define i1 @test_has_sse2() {
; HASWELL-LABEL: @test_has_sse2(
; HASWELL-NEXT:    ret i1 true
;
; GENERIC-LABEL: @test_has_sse2(
; GENERIC-NEXT:    ret i1 true
  %1 = call i1 @llvm.target.has.feature(metadata !"sse2")
  ret i1 %1
}

define i1 @test_has_bogus() {
; HASWELL-LABEL: @test_has_bogus(
; HASWELL-NEXT:    ret i1 false
;
; GENERIC-LABEL: @test_has_bogus(
; GENERIC-NEXT:    ret i1 false
  %1 = call i1 @llvm.target.has.feature(metadata !"bogus")
  ret i1 %1
}

define i1 @test_is_haswell() {
; HASWELL-LABEL: @test_is_haswell(
; HASWELL-NEXT:    ret i1 true
;
; GENERIC-LABEL: @test_is_haswell(
; GENERIC-NEXT:    ret i1 false
  %1 = call i1 @llvm.target.is.cpu(metadata !"haswell")
  ret i1 %1
}

define i1 @test_is_generic() {
; HASWELL-LABEL: @test_is_generic(
; HASWELL-NEXT:    ret i1 false
;
; GENERIC-LABEL: @test_is_generic(
; GENERIC-NEXT:    ret i1 true
  %1 = call i1 @llvm.target.is.cpu(metadata !"x86-64")
  ret i1 %1
}
