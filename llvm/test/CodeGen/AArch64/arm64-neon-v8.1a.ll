; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -aarch64-neon-syntax=generic | FileCheck %s --check-prefix=CHECK-V8a
; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -mattr=+rdm -aarch64-neon-syntax=generic | FileCheck %s --check-prefix=CHECK-V81a
; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -mcpu=falkor -aarch64-neon-syntax=generic | FileCheck %s --check-prefix=CHECK-V81a
; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -mattr=+v8.1a -aarch64-neon-syntax=generic | FileCheck %s --check-prefix=CHECK-V81a
; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -mattr=+v8.1a -aarch64-neon-syntax=apple | FileCheck %s --check-prefix=CHECK-V81a-apple

declare <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32>, <4 x i32>)
declare i32 @llvm.aarch64.neon.sqrdmulh.i32(i32, i32)
declare i16 @llvm.aarch64.neon.sqrdmulh.i16(i16, i16)

declare <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32>, <4 x i32>)
declare i32 @llvm.aarch64.neon.sqadd.i32(i32, i32)
declare i16 @llvm.aarch64.neon.sqadd.i16(i16, i16)

declare <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32>, <4 x i32>)
declare i32 @llvm.aarch64.neon.sqsub.i32(i32, i32)
declare i16 @llvm.aarch64.neon.sqsub.i16(i16, i16)

;-----------------------------------------------------------------------------
; RDMA Vector
; test for SIMDThreeSameVectorSQRDMLxHTiedHS

define <4 x i16> @test_sqrdmlah_v4i16(<4 x i16> %acc, <4 x i16> %mhs, <4 x i16> %rhs) {
; CHECK-LABEL: test_sqrdmlah_v4i16:
   %prod = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %mhs,  <4 x i16> %rhs)
   %retval =  call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> %acc,  <4 x i16> %prod)
; CHECK-V8a:        sqrdmulh    v1.4h, v1.4h, v2.4h
; CHECK-V81a:       sqrdmlah    v0.4h, v1.4h, v2.4h
; CHECK-V81a-apple: sqrdmlah.4h v0,    v1,    v2
   ret <4 x i16> %retval
}

define <8 x i16> @test_sqrdmlah_v8i16(<8 x i16> %acc, <8 x i16> %mhs, <8 x i16> %rhs) {
; CHECK-LABEL: test_sqrdmlah_v8i16:
   %prod = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %mhs, <8 x i16> %rhs)
   %retval =  call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> %acc, <8 x i16> %prod)
; CHECK-V8a:        sqrdmulh    v1.8h, v1.8h, v2.8h
; CHECK-V81a:       sqrdmlah    v0.8h, v1.8h, v2.8h
; CHECK-V81a-apple: sqrdmlah.8h v0, v1, v2
   ret <8 x i16> %retval
}

define <2 x i32> @test_sqrdmlah_v2i32(<2 x i32> %acc, <2 x i32> %mhs, <2 x i32> %rhs) {
; CHECK-LABEL: test_sqrdmlah_v2i32:
   %prod = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %mhs, <2 x i32> %rhs)
   %retval =  call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> %acc, <2 x i32> %prod)
; CHECK-V8a:        sqrdmulh    v1.2s, v1.2s, v2.2s
; CHECK-V81a:       sqrdmlah    v0.2s, v1.2s, v2.2s
; CHECK-V81a-apple: sqrdmlah.2s v0,    v1,    v2
   ret <2 x i32> %retval
}

define <4 x i32> @test_sqrdmlah_v4i32(<4 x i32> %acc, <4 x i32> %mhs, <4 x i32> %rhs) {
; CHECK-LABEL: test_sqrdmlah_v4i32:
   %prod = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %mhs, <4 x i32> %rhs)
   %retval =  call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %acc, <4 x i32> %prod)
; CHECK-V81:        sqrdmulh    v1.4s, v1.4s, v2.4s
; CHECK-V81a:       sqrdmlah    v0.4s, v1.4s, v2.4s
; CHECK-V81a-apple: sqrdmlah.4s v0,    v1,    v2
   ret <4 x i32> %retval
}

define <4 x i16> @test_sqrdmlsh_v4i16(<4 x i16> %acc, <4 x i16> %mhs, <4 x i16> %rhs) {
; CHECK-LABEL: test_sqrdmlsh_v4i16:
   %prod = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %mhs,  <4 x i16> %rhs)
   %retval =  call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> %acc, <4 x i16> %prod)
; CHECK-V8a:        sqrdmulh    v1.4h, v1.4h, v2.4h
; CHECK-V81a:       sqrdmlsh    v0.4h, v1.4h, v2.4h
; CHECK-V81a-apple: sqrdmlsh.4h v0,    v1,    v2
   ret <4 x i16> %retval
}

define <8 x i16> @test_sqrdmlsh_v8i16(<8 x i16> %acc, <8 x i16> %mhs, <8 x i16> %rhs) {
; CHECK-LABEL: test_sqrdmlsh_v8i16:
   %prod = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %mhs, <8 x i16> %rhs)
   %retval =  call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> %acc, <8 x i16> %prod)
; CHECK-V8a:        sqrdmulh    v1.8h, v1.8h, v2.8h
; CHECK-V81a:       sqrdmlsh    v0.8h, v1.8h, v2.8h
; CHECK-V81a-apple: sqrdmlsh.8h v0,    v1,    v2
   ret <8 x i16> %retval
}

define <2 x i32> @test_sqrdmlsh_v2i32(<2 x i32> %acc, <2 x i32> %mhs, <2 x i32> %rhs) {
; CHECK-LABEL: test_sqrdmlsh_v2i32:
   %prod = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %mhs, <2 x i32> %rhs)
   %retval =  call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> %acc, <2 x i32> %prod)
; CHECK-V8a:        sqrdmulh    v1.2s, v1.2s, v2.2s
; CHECK-V81a:       sqrdmlsh    v0.2s, v1.2s, v2.2s
; CHECK-V81a-apple: sqrdmlsh.2s v0,    v1,    v2
   ret <2 x i32> %retval
}

define <4 x i32> @test_sqrdmlsh_v4i32(<4 x i32> %acc, <4 x i32> %mhs, <4 x i32> %rhs) {
; CHECK-LABEL: test_sqrdmlsh_v4i32:
   %prod = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %mhs, <4 x i32> %rhs)
   %retval =  call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %acc, <4 x i32> %prod)
; CHECK-V8a:        sqrdmulh    v1.4s, v1.4s, v2.4s
; CHECK-V81a:       sqrdmlsh    v0.4s, v1.4s, v2.4s
; CHECK-V81a-apple: sqrdmlsh.4s v0,    v1,    v2
   ret <4 x i32> %retval
}

;-----------------------------------------------------------------------------
; RDMA Vector, by element
; tests for vXiYY_indexed in SIMDIndexedSQRDMLxHSDTied

define <4 x i16> @test_sqrdmlah_lane_s16(<4 x i16> %acc, <4 x i16> %x, <4 x i16> %v) {
; CHECK-LABEL: test_sqrdmlah_lane_s16:
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %prod = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %x, <4 x i16> %shuffle)
  %retval =  call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> %acc, <4 x i16> %prod)
; CHECK-V8a :       sqrdmulh    v1.4h, v1.4h, v2.h[3]
; CHECK-V81a:       sqrdmlah    v0.4h, v1.4h, v2.h[3]
; CHECK-V81a-apple: sqrdmlah.4h v0,    v1,    v2[3]
  ret <4 x i16> %retval
}

define <8 x i16> @test_sqrdmlahq_lane_s16(<8 x i16> %acc, <8 x i16> %x, <8 x i16> %v) {
; CHECK-LABEL: test_sqrdmlahq_lane_s16:
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %prod = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %x, <8 x i16> %shuffle)
  %retval =  call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> %acc, <8 x i16> %prod)
; CHECK-V8a:        sqrdmulh    v1.8h, v1.8h, v2.h[2]
; CHECK-V81a:       sqrdmlah    v0.8h, v1.8h, v2.h[2]
; CHECK-V81a-apple: sqrdmlah.8h v0,    v1,    v2[2]
  ret <8 x i16> %retval
}

define <2 x i32> @test_sqrdmlah_lane_s32(<2 x i32> %acc, <2 x i32> %x, <2 x i32> %v) {
; CHECK-LABEL: test_sqrdmlah_lane_s32:
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %prod = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %x, <2 x i32> %shuffle)
  %retval =  call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> %acc, <2 x i32> %prod)
; CHECK-V8a:        sqrdmulh    v1.2s, v1.2s, v2.s[1]
; CHECK-V81a:       sqrdmlah    v0.2s, v1.2s, v2.s[1]
; CHECK-V81a-apple: sqrdmlah.2s v0,    v1,    v2[1]
  ret <2 x i32> %retval
}

define <4 x i32> @test_sqrdmlahq_lane_s32(<4 x i32> %acc,<4 x i32> %x, <4 x i32> %v) {
; CHECK-LABEL: test_sqrdmlahq_lane_s32:
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> zeroinitializer
  %prod = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %x, <4 x i32> %shuffle)
  %retval =  call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %acc, <4 x i32> %prod)
; CHECK-V8a:        sqrdmulh    v1.4s, v1.4s, v2.s[0]
; CHECK-V81a:       sqrdmlah    v0.4s, v1.4s, v2.s[0]
; CHECK-V81a-apple: sqrdmlah.4s v0,    v1,    v2[0]
  ret <4 x i32> %retval
}

define <4 x i16> @test_sqrdmlsh_lane_s16(<4 x i16> %acc, <4 x i16> %x, <4 x i16> %v) {
; CHECK-LABEL: test_sqrdmlsh_lane_s16:
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %prod = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %x, <4 x i16> %shuffle)
  %retval =  call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> %acc, <4 x i16> %prod)
; CHECK-V8a:        sqrdmulh    v1.4h, v1.4h, v2.h[3]
; CHECK-V81a:       sqrdmlsh    v0.4h, v1.4h, v2.h[3]
; CHECK-V81a-apple: sqrdmlsh.4h v0,    v1,    v2[3]
  ret <4 x i16> %retval
}

define <8 x i16> @test_sqrdmlshq_lane_s16(<8 x i16> %acc, <8 x i16> %x, <8 x i16> %v) {
; CHECK-LABEL: test_sqrdmlshq_lane_s16:
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %prod = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %x, <8 x i16> %shuffle)
  %retval =  call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> %acc, <8 x i16> %prod)
; CHECK-V8a:        sqrdmulh    v1.8h, v1.8h, v2.h[2]
; CHECK-V81a:       sqrdmlsh    v0.8h, v1.8h, v2.h[2]
; CHECK-V81a-apple: sqrdmlsh.8h v0,    v1,    v2[2]
  ret <8 x i16> %retval
}

define <2 x i32> @test_sqrdmlsh_lane_s32(<2 x i32> %acc, <2 x i32> %x, <2 x i32> %v) {
; CHECK-LABEL: test_sqrdmlsh_lane_s32:
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %prod = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %x, <2 x i32> %shuffle)
  %retval =  call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> %acc, <2 x i32> %prod)
; CHECK-V8a:        sqrdmulh    v1.2s, v1.2s, v2.s[1]
; CHECK-V81a:       sqrdmlsh    v0.2s, v1.2s, v2.s[1]
; CHECK-V81a-apple: sqrdmlsh.2s v0,    v1,    v2[1]
  ret <2 x i32> %retval
}

define <4 x i32> @test_sqrdmlshq_lane_s32(<4 x i32> %acc,<4 x i32> %x, <4 x i32> %v) {
; CHECK-LABEL: test_sqrdmlshq_lane_s32:
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> zeroinitializer
  %prod = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %x, <4 x i32> %shuffle)
  %retval =  call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %acc, <4 x i32> %prod)
; CHECK-V8a:        sqrdmulh    v1.4s, v1.4s, v2.s[0]
; CHECK-V81a:       sqrdmlsh    v0.4s, v1.4s, v2.s[0]
; CHECK-V81a-apple: sqrdmlsh.4s v0,    v1,    v2[0]
  ret <4 x i32> %retval
}

;-----------------------------------------------------------------------------
; RDMA Vector, by element, extracted
; i16 tests are for vXi16_indexed in SIMDIndexedSQRDMLxHSDTied, with IR in ACLE style
; i32 tests are for   "def : Pat" in SIMDIndexedSQRDMLxHSDTied

define i16 @test_sqrdmlah_extracted_lane_s16(i16 %acc,<4 x i16> %x, <4 x i16> %v) {
; CHECK-LABEL: test_sqrdmlah_extracted_lane_s16:
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 1,i32 1,i32 1,i32 1>
  %prod = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %x, <4 x i16> %shuffle)
  %acc_vec = insertelement <4 x i16> undef, i16 %acc, i64 0
  %retval_vec =  call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> %acc_vec, <4 x i16> %prod)
  %retval = extractelement <4 x i16> %retval_vec, i64 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.4h, v0.4h, v1.h[1]
; CHECK-V81a:       sqrdmlah    {{v[2-9]+}}.4h, v0.4h, v1.h[1]
; CHECK-V81a-apple: sqrdmlah.4h {{v[2-9]+}},    v0,    v1[1]
  ret i16 %retval
}

define i16 @test_sqrdmlahq_extracted_lane_s16(i16 %acc,<8 x i16> %x, <8 x i16> %v) {
; CHECK-LABEL: test_sqrdmlahq_extracted_lane_s16:
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 1,i32 1,i32 1,i32 1, i32 1,i32 1,i32 1,i32 1>
  %prod = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %x, <8 x i16> %shuffle)
  %acc_vec = insertelement <8 x i16> undef, i16 %acc, i64 0
  %retval_vec =  call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> %acc_vec, <8 x i16> %prod)
  %retval = extractelement <8 x i16> %retval_vec, i64 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.8h, v0.8h, v1.h[1]
; CHECK-V81a:       sqrdmlah    {{v[2-9]+}}.8h, v0.8h, v1.h[1]
; CHECK-V81a-apple: sqrdmlah.8h {{v[2-9]+}},    v0,    v1[1]
  ret i16 %retval
}

define i32 @test_sqrdmlah_extracted_lane_s32(i32 %acc,<2 x i32> %x, <2 x i32> %v) {
; CHECK-LABEL: test_sqrdmlah_extracted_lane_s32:
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %prod = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %x, <2 x i32> %shuffle)
  %extract = extractelement <2 x i32> %prod, i64 0
  %retval =  call i32 @llvm.aarch64.neon.sqadd.i32(i32 %acc, i32 %extract)
; CHECK-V8a:        sqrdmulh    v0.2s, v0.2s, v1.s[0]
; CHECK-V81a:       sqrdmlah    v2.2s, v0.2s, v1.s[0]
; CHECK-V81a-apple: sqrdmlah.2s v2,    v0,    v1[0]
  ret i32 %retval
}

define i32 @test_sqrdmlahq_extracted_lane_s32(i32 %acc,<4 x i32> %x, <4 x i32> %v) {
; CHECK-LABEL: test_sqrdmlahq_extracted_lane_s32:
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> zeroinitializer
  %prod = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %x, <4 x i32> %shuffle)
  %extract = extractelement <4 x i32> %prod, i64 0
  %retval =  call i32 @llvm.aarch64.neon.sqadd.i32(i32 %acc, i32 %extract)
; CHECK-V8a:        sqrdmulh    v0.4s, v0.4s, v1.s[0]
; CHECK-V81a:       sqrdmlah    v2.4s, v0.4s, v1.s[0]
; CHECK-V81a-apple: sqrdmlah.4s v2,    v0,    v1[0]
  ret i32 %retval
}

define i16 @test_sqrdmlsh_extracted_lane_s16(i16 %acc,<4 x i16> %x, <4 x i16> %v) {
; CHECK-LABEL: test_sqrdmlsh_extracted_lane_s16:
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 1,i32 1,i32 1,i32 1>
  %prod = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %x, <4 x i16> %shuffle)
  %acc_vec = insertelement <4 x i16> undef, i16 %acc, i64 0
  %retval_vec =  call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> %acc_vec, <4 x i16> %prod)
  %retval = extractelement <4 x i16> %retval_vec, i64 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.4h, v0.4h, v1.h[1]
; CHECK-V81a:       sqrdmlsh    {{v[2-9]+}}.4h, v0.4h, v1.h[1]
; CHECK-V81a-apple: sqrdmlsh.4h {{v[2-9]+}},    v0,    v1[1]
  ret i16 %retval
}

define i16 @test_sqrdmlshq_extracted_lane_s16(i16 %acc,<8 x i16> %x, <8 x i16> %v) {
; CHECK-LABEL: test_sqrdmlshq_extracted_lane_s16:
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 1,i32 1,i32 1,i32 1, i32 1,i32 1,i32 1,i32 1>
  %prod = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %x, <8 x i16> %shuffle)
  %acc_vec = insertelement <8 x i16> undef, i16 %acc, i64 0
  %retval_vec =  call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> %acc_vec, <8 x i16> %prod)
  %retval = extractelement <8 x i16> %retval_vec, i64 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.8h, v0.8h, v1.h[1]
; CHECK-V81a:       sqrdmlsh    {{v[2-9]+}}.8h, v0.8h, v1.h[1]
; CHECK-V81a-apple: sqrdmlsh.8h {{v[2-9]+}},    v0,    v1[1]
  ret i16 %retval
}

define i32 @test_sqrdmlsh_extracted_lane_s32(i32 %acc,<2 x i32> %x, <2 x i32> %v) {
; CHECK-LABEL: test_sqrdmlsh_extracted_lane_s32:
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %prod = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %x, <2 x i32> %shuffle)
  %extract = extractelement <2 x i32> %prod, i64 0
  %retval =  call i32 @llvm.aarch64.neon.sqsub.i32(i32 %acc, i32 %extract)
; CHECK-V8a:        sqrdmulh    v0.2s, v0.2s, v1.s[0]
; CHECK-V81a:       sqrdmlsh    v2.2s, v0.2s, v1.s[0]
; CHECK-V81a-apple: sqrdmlsh.2s v2,    v0,    v1[0]
  ret i32 %retval
}

define i32 @test_sqrdmlshq_extracted_lane_s32(i32 %acc,<4 x i32> %x, <4 x i32> %v) {
; CHECK-LABEL: test_sqrdmlshq_extracted_lane_s32:
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> zeroinitializer
  %prod = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %x, <4 x i32> %shuffle)
  %extract = extractelement <4 x i32> %prod, i64 0
  %retval =  call i32 @llvm.aarch64.neon.sqsub.i32(i32 %acc, i32 %extract)
; CHECK-V8a:        sqrdmulh    v0.4s, v0.4s, v1.s[0]
; CHECK-V81a:       sqrdmlsh    v2.4s, v0.4s, v1.s[0]
; CHECK-V81a-apple: sqrdmlsh.4s v2,    v0,    v1[0]
  ret i32 %retval
}

;-----------------------------------------------------------------------------
; RDMA Scalar
; test for "def : Pat" near SIMDThreeScalarHSTied in AArch64InstInfo.td

define i16 @test_sqrdmlah_v1i16(i16 %acc, i16 %x, i16 %y) {
; CHECK-LABEL: test_sqrdmlah_v1i16:
  %x_vec = insertelement <4 x i16> undef, i16 %x, i64 0
  %y_vec = insertelement <4 x i16> undef, i16 %y, i64 0
  %prod_vec = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %x_vec,  <4 x i16> %y_vec)
  %acc_vec = insertelement <4 x i16> undef, i16 %acc, i64 0
  %retval_vec =  call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> %acc_vec,  <4 x i16> %prod_vec)
  %retval = extractelement <4 x i16> %retval_vec, i64 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK-V81a:       sqrdmlah    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK-V81a-apple: sqrdmlah.4h {{v[0-9]+}},    {{v[0-9]+}},    {{v[0-9]+}}
  ret i16 %retval
}

define i32 @test_sqrdmlah_v1i32(i32 %acc, i32 %x, i32 %y) {
; CHECK-LABEL: test_sqrdmlah_v1i32:
  %x_vec = insertelement <4 x i32> undef, i32 %x, i64 0
  %y_vec = insertelement <4 x i32> undef, i32 %y, i64 0
  %prod_vec = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %x_vec,  <4 x i32> %y_vec)
  %acc_vec = insertelement <4 x i32> undef, i32 %acc, i64 0
  %retval_vec =  call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %acc_vec,  <4 x i32> %prod_vec)
  %retval = extractelement <4 x i32> %retval_vec, i64 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-V81a:       sqrdmlah    {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-V81a-apple: sqrdmlah.4s {{v[0-9]+}},    {{v[0-9]+}},    {{v[0-9]+}}
  ret i32 %retval
}


define i16 @test_sqrdmlsh_v1i16(i16 %acc, i16 %x, i16 %y) {
; CHECK-LABEL: test_sqrdmlsh_v1i16:
  %x_vec = insertelement <4 x i16> undef, i16 %x, i64 0
  %y_vec = insertelement <4 x i16> undef, i16 %y, i64 0
  %prod_vec = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %x_vec,  <4 x i16> %y_vec)
  %acc_vec = insertelement <4 x i16> undef, i16 %acc, i64 0
  %retval_vec =  call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> %acc_vec,  <4 x i16> %prod_vec)
  %retval = extractelement <4 x i16> %retval_vec, i64 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK-V81a:       sqrdmlsh    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK-V81a-apple: sqrdmlsh.4h {{v[0-9]+}},    {{v[0-9]+}},    {{v[0-9]+}}
  ret i16 %retval
}

define i32 @test_sqrdmlsh_v1i32(i32 %acc, i32 %x, i32 %y) {
; CHECK-LABEL: test_sqrdmlsh_v1i32:
  %x_vec = insertelement <4 x i32> undef, i32 %x, i64 0
  %y_vec = insertelement <4 x i32> undef, i32 %y, i64 0
  %prod_vec = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %x_vec,  <4 x i32> %y_vec)
  %acc_vec = insertelement <4 x i32> undef, i32 %acc, i64 0
  %retval_vec =  call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %acc_vec,  <4 x i32> %prod_vec)
  %retval = extractelement <4 x i32> %retval_vec, i64 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-V81a:       sqrdmlsh    {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-V81a-apple: sqrdmlsh.4s {{v[0-9]+}},    {{v[0-9]+}},    {{v[0-9]+}}
  ret i32 %retval
}
define i32 @test_sqrdmlah_i32(i32 %acc, i32 %mhs, i32 %rhs) {
; CHECK-LABEL: test_sqrdmlah_i32:
  %prod = call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 %mhs,  i32 %rhs)
  %retval =  call i32 @llvm.aarch64.neon.sqadd.i32(i32 %acc,  i32 %prod)
; CHECK-V8a:        sqrdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-V81a:       sqrdmlah {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-V81a-apple: sqrdmlah {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret i32 %retval
}

define i32 @test_sqrdmlsh_i32(i32 %acc, i32 %mhs, i32 %rhs) {
; CHECK-LABEL: test_sqrdmlsh_i32:
  %prod = call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 %mhs,  i32 %rhs)
  %retval =  call i32 @llvm.aarch64.neon.sqsub.i32(i32 %acc,  i32 %prod)
; CHECK-V8a:        sqrdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-V81a:       sqrdmlsh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-V81a-apple: sqrdmlsh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret i32 %retval
}

;-----------------------------------------------------------------------------
; RDMA Scalar, by element
; i16 tests are performed via tests in above chapter, with IR in ACLE style
; i32 tests are for i32_indexed in SIMDIndexedSQRDMLxHSDTied

define i16 @test_sqrdmlah_extract_i16(i16 %acc, i16 %x, <4 x i16> %y_vec) {
; CHECK-LABEL: test_sqrdmlah_extract_i16:
  %shuffle = shufflevector <4 x i16> %y_vec, <4 x i16> undef, <4 x i32> <i32 1,i32 1,i32 1,i32 1>
  %x_vec = insertelement <4 x i16> undef, i16 %x, i64 0
  %prod = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %x_vec, <4 x i16> %shuffle)
  %acc_vec = insertelement <4 x i16> undef, i16 %acc, i64 0
  %retval_vec =  call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> %acc_vec, <4 x i16> %prod)
  %retval = extractelement <4 x i16> %retval_vec, i32 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, v0.h[1]
; CHECK-V81a:       sqrdmlah    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, v0.h[1]
; CHECK-V81a-apple: sqrdmlah.4h {{v[0-9]+}},    {{v[0-9]+}}, v0[1]
  ret i16 %retval
}

define i32 @test_sqrdmlah_extract_i32(i32 %acc, i32 %mhs, <4 x i32> %rhs) {
; CHECK-LABEL: test_sqrdmlah_extract_i32:
  %extract = extractelement <4 x i32> %rhs, i32 3
  %prod = call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 %mhs,  i32 %extract)
  %retval =  call i32 @llvm.aarch64.neon.sqadd.i32(i32 %acc,  i32 %prod)
; CHECK-V8a:        sqrdmulh   {{s[0-9]+}}, {{s[0-9]+}}, v0.s[3]
; CHECK-V81a:       sqrdmlah   {{s[0-9]+}}, {{s[0-9]+}}, v0.s[3]
; CHECK-V81a-apple: sqrdmlah.s {{s[0-9]+}}, {{s[0-9]+}}, v0[3]
  ret i32 %retval
}

define i16 @test_sqrdmlshq_extract_i16(i16 %acc, i16 %x, <8 x i16> %y_vec) {
; CHECK-LABEL: test_sqrdmlshq_extract_i16:
  %shuffle = shufflevector <8 x i16> %y_vec, <8 x i16> undef, <8 x i32> <i32 1,i32 1,i32 1,i32 1,i32 1,i32 1,i32 1,i32 1>
  %x_vec = insertelement <8 x i16> undef, i16 %x, i64 0
  %prod = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %x_vec, <8 x i16> %shuffle)
  %acc_vec = insertelement <8 x i16> undef, i16 %acc, i64 0
  %retval_vec =  call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> %acc_vec, <8 x i16> %prod)
  %retval = extractelement <8 x i16> %retval_vec, i32 0
; CHECK-V8a:        sqrdmulh    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, v0.h[1]
; CHECK-V81a:       sqrdmlsh    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, v0.h[1]
; CHECK-V81a-apple: sqrdmlsh.8h {{v[0-9]+}},    {{v[0-9]+}}, v0[1]
  ret i16 %retval
}

define i32 @test_sqrdmlsh_extract_i32(i32 %acc, i32 %mhs, <4 x i32> %rhs) {
; CHECK-LABEL: test_sqrdmlsh_extract_i32:
  %extract = extractelement <4 x i32> %rhs, i32 3
  %prod = call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 %mhs,  i32 %extract)
  %retval =  call i32 @llvm.aarch64.neon.sqsub.i32(i32 %acc,  i32 %prod)
; CHECK-V8a:        sqrdmulh   {{s[0-9]+}}, {{s[0-9]+}}, v0.s[3]
; CHECK-V81a:       sqrdmlsh   {{s[0-9]+}}, {{s[0-9]+}}, v0.s[3]
; CHECK-V81a-apple: sqrdmlsh.s {{s[0-9]+}}, {{s[0-9]+}}, v0[3]
  ret i32 %retval
}
