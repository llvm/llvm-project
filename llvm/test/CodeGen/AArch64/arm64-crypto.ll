; RUN: llc -mtriple=arm64-eabi -mattr=crypto -aarch64-neon-syntax=apple -o - %s | FileCheck %s

declare <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %data, <16 x i8> %key)
declare <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %data, <16 x i8> %key)
declare <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %data)
declare <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %data)

define <16 x i8> @test_aese(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: test_aese:
; CHECK: aese.16b v0, v1
  %res = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %data, <16 x i8> %key)
  ret <16 x i8> %res
}

define <16 x i8> @test_aesd(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: test_aesd:
; CHECK: aesd.16b v0, v1
  %res = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %data, <16 x i8> %key)
  ret <16 x i8> %res
}

define <16 x i8> @test_aesmc(<16 x i8> %data) {
; CHECK-LABEL: test_aesmc:
; CHECK: aesmc.16b v0, v0
 %res = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %data)
  ret <16 x i8> %res
}

define <16 x i8> @test_aesimc(<16 x i8> %data) {
; CHECK-LABEL: test_aesimc:
; CHECK: aesimc.16b v0, v0
 %res = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %data)
  ret <16 x i8> %res
}

declare <4 x i32> @llvm.aarch64.crypto.sha1c(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
declare <4 x i32> @llvm.aarch64.crypto.sha1p(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
declare <4 x i32> @llvm.aarch64.crypto.sha1m(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
declare i32 @llvm.aarch64.crypto.sha1h(i32 %hash_e)
declare <4 x i32> @llvm.aarch64.crypto.sha1su0(<4 x i32> %wk0_3, <4 x i32> %wk4_7, <4 x i32> %wk8_11)
declare <4 x i32> @llvm.aarch64.crypto.sha1su1(<4 x i32> %wk0_3, <4 x i32> %wk12_15)

define <4 x i32> @test_sha1c(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk) {
; CHECK-LABEL: test_sha1c:
; CHECK: fmov [[HASH_E:s[0-9]+]], w0
; CHECK: sha1c.4s q0, [[HASH_E]], v1
  %res = call <4 x i32> @llvm.aarch64.crypto.sha1c(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
  ret <4 x i32> %res
}

; <rdar://problem/14742333> Incomplete removal of unnecessary FMOV instructions in intrinsic SHA1
define <4 x i32> @test_sha1c_in_a_row(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk) {
; CHECK-LABEL: test_sha1c_in_a_row:
; CHECK: fmov [[HASH_E:s[0-9]+]], w0
; CHECK: sha1c.4s q[[SHA1RES:[0-9]+]], [[HASH_E]], v1
; CHECK-NOT: fmov
; CHECK: sha1c.4s q0, s[[SHA1RES]], v1
  %res = call <4 x i32> @llvm.aarch64.crypto.sha1c(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
  %extract = extractelement <4 x i32> %res, i32 0
  %res2 = call <4 x i32> @llvm.aarch64.crypto.sha1c(<4 x i32> %hash_abcd, i32 %extract, <4 x i32> %wk)
  ret <4 x i32> %res2
}

define <4 x i32> @test_sha1p(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk) {
; CHECK-LABEL: test_sha1p:
; CHECK: fmov [[HASH_E:s[0-9]+]], w0
; CHECK: sha1p.4s q0, [[HASH_E]], v1
  %res = call <4 x i32> @llvm.aarch64.crypto.sha1p(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
  ret <4 x i32> %res
}

define <4 x i32> @test_sha1m(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk) {
; CHECK-LABEL: test_sha1m:
; CHECK: fmov [[HASH_E:s[0-9]+]], w0
; CHECK: sha1m.4s q0, [[HASH_E]], v1
  %res = call <4 x i32> @llvm.aarch64.crypto.sha1m(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
  ret <4 x i32> %res
}

define i32 @test_sha1h(i32 %hash_e) {
; CHECK-LABEL: test_sha1h:
; CHECK: fmov [[HASH_E:s[0-9]+]], w0
; CHECK: sha1h [[RES:s[0-9]+]], [[HASH_E]]
; CHECK: fmov w0, [[RES]]
  %res = call i32 @llvm.aarch64.crypto.sha1h(i32 %hash_e)
  ret i32 %res
}

define <4 x i32> @test_sha1su0(<4 x i32> %wk0_3, <4 x i32> %wk4_7, <4 x i32> %wk8_11) {
; CHECK-LABEL: test_sha1su0:
; CHECK: sha1su0.4s v0, v1, v2
  %res = call <4 x i32> @llvm.aarch64.crypto.sha1su0(<4 x i32> %wk0_3, <4 x i32> %wk4_7, <4 x i32> %wk8_11)
  ret <4 x i32> %res
}

define <4 x i32> @test_sha1su1(<4 x i32> %wk0_3, <4 x i32> %wk12_15) {
; CHECK-LABEL: test_sha1su1:
; CHECK: sha1su1.4s v0, v1
  %res = call <4 x i32> @llvm.aarch64.crypto.sha1su1(<4 x i32> %wk0_3, <4 x i32> %wk12_15)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.aarch64.crypto.sha256h(<4 x i32> %hash_abcd, <4 x i32> %hash_efgh, <4 x i32> %wk)
declare <4 x i32> @llvm.aarch64.crypto.sha256h2(<4 x i32> %hash_efgh, <4 x i32> %hash_abcd, <4 x i32> %wk)
declare <4 x i32> @llvm.aarch64.crypto.sha256su0(<4 x i32> %w0_3, <4 x i32> %w4_7)
declare <4 x i32> @llvm.aarch64.crypto.sha256su1(<4 x i32> %w0_3, <4 x i32> %w8_11, <4 x i32> %w12_15)

define <4 x i32> @test_sha256h(<4 x i32> %hash_abcd, <4 x i32> %hash_efgh, <4 x i32> %wk) {
; CHECK-LABEL: test_sha256h:
; CHECK: sha256h.4s q0, q1, v2
  %res = call <4 x i32> @llvm.aarch64.crypto.sha256h(<4 x i32> %hash_abcd, <4 x i32> %hash_efgh, <4 x i32> %wk)
  ret <4 x i32> %res
}

define <4 x i32> @test_sha256h2(<4 x i32> %hash_efgh, <4 x i32> %hash_abcd, <4 x i32> %wk) {
; CHECK-LABEL: test_sha256h2:
; CHECK: sha256h2.4s q0, q1, v2

  %res = call <4 x i32> @llvm.aarch64.crypto.sha256h2(<4 x i32> %hash_efgh, <4 x i32> %hash_abcd, <4 x i32> %wk)
  ret <4 x i32> %res
}

define <4 x i32> @test_sha256su0(<4 x i32> %w0_3, <4 x i32> %w4_7) {
; CHECK-LABEL: test_sha256su0:
; CHECK: sha256su0.4s v0, v1
  %res = call <4 x i32> @llvm.aarch64.crypto.sha256su0(<4 x i32> %w0_3, <4 x i32> %w4_7)
  ret <4 x i32> %res
}

define <4 x i32> @test_sha256su1(<4 x i32> %w0_3, <4 x i32> %w8_11, <4 x i32> %w12_15) {
; CHECK-LABEL: test_sha256su1:
; CHECK: sha256su1.4s v0, v1, v2
  %res = call <4 x i32> @llvm.aarch64.crypto.sha256su1(<4 x i32> %w0_3, <4 x i32> %w8_11, <4 x i32> %w12_15)
  ret <4 x i32> %res
}
