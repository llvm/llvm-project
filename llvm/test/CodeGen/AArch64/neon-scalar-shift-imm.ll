; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define i64 @test_vshrd_n_s64(i64 %a) {
; CHECK: test_vshrd_n_s64
; CHECK: sshr {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsshr = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsshr1 = call <1 x i64> @llvm.aarch64.neon.vshrds.n(<1 x i64> %vsshr, i32 63)
  %0 = extractelement <1 x i64> %vsshr1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vshrds.n(<1 x i64>, i32)

define i64 @test_vshrd_n_u64(i64 %a) {
; CHECK: test_vshrd_n_u64
; CHECK: ushr {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vushr = insertelement <1 x i64> undef, i64 %a, i32 0
  %vushr1 = call <1 x i64> @llvm.aarch64.neon.vshrdu.n(<1 x i64> %vushr, i32 63)
  %0 = extractelement <1 x i64> %vushr1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vshrdu.n(<1 x i64>, i32)

define i64 @test_vrshrd_n_s64(i64 %a) {
; CHECK: test_vrshrd_n_s64
; CHECK: srshr {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsrshr = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsrshr1 = call <1 x i64> @llvm.aarch64.neon.vsrshr.v1i64(<1 x i64> %vsrshr, i32 63)
  %0 = extractelement <1 x i64> %vsrshr1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vsrshr.v1i64(<1 x i64>, i32)

define i64 @test_vrshrd_n_u64(i64 %a) {
; CHECK: test_vrshrd_n_u64
; CHECK: urshr {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vurshr = insertelement <1 x i64> undef, i64 %a, i32 0
  %vurshr1 = call <1 x i64> @llvm.aarch64.neon.vurshr.v1i64(<1 x i64> %vurshr, i32 63)
  %0 = extractelement <1 x i64> %vurshr1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vurshr.v1i64(<1 x i64>, i32)

define i64 @test_vsrad_n_s64(i64 %a, i64 %b) {
; CHECK: test_vsrad_n_s64
; CHECK: ssra {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vssra = insertelement <1 x i64> undef, i64 %a, i32 0
  %vssra1 = insertelement <1 x i64> undef, i64 %b, i32 0
  %vssra2 = call <1 x i64> @llvm.aarch64.neon.vsrads.n(<1 x i64> %vssra, <1 x i64> %vssra1, i32 63)
  %0 = extractelement <1 x i64> %vssra2, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vsrads.n(<1 x i64>, <1 x i64>, i32)

define i64 @test_vsrad_n_u64(i64 %a, i64 %b) {
; CHECK: test_vsrad_n_u64
; CHECK: usra {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vusra = insertelement <1 x i64> undef, i64 %a, i32 0
  %vusra1 = insertelement <1 x i64> undef, i64 %b, i32 0
  %vusra2 = call <1 x i64> @llvm.aarch64.neon.vsradu.n(<1 x i64> %vusra, <1 x i64> %vusra1, i32 63)
  %0 = extractelement <1 x i64> %vusra2, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vsradu.n(<1 x i64>, <1 x i64>, i32)

define i64 @test_vrsrad_n_s64(i64 %a, i64 %b) {
; CHECK: test_vrsrad_n_s64
; CHECK: srsra {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsrsra = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsrsra1 = insertelement <1 x i64> undef, i64 %b, i32 0
  %vsrsra2 = call <1 x i64> @llvm.aarch64.neon.vrsrads.n(<1 x i64> %vsrsra, <1 x i64> %vsrsra1, i32 63)
  %0 = extractelement <1 x i64> %vsrsra2, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vrsrads.n(<1 x i64>, <1 x i64>, i32)

define i64 @test_vrsrad_n_u64(i64 %a, i64 %b) {
; CHECK: test_vrsrad_n_u64
; CHECK: ursra {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vursra = insertelement <1 x i64> undef, i64 %a, i32 0
  %vursra1 = insertelement <1 x i64> undef, i64 %b, i32 0
  %vursra2 = call <1 x i64> @llvm.aarch64.neon.vrsradu.n(<1 x i64> %vursra, <1 x i64> %vursra1, i32 63)
  %0 = extractelement <1 x i64> %vursra2, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vrsradu.n(<1 x i64>, <1 x i64>, i32)

define i64 @test_vshld_n_s64(i64 %a) {
; CHECK: test_vshld_n_s64
; CHECK: shl {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vshl = insertelement <1 x i64> undef, i64 %a, i32 0
  %vshl1 = call <1 x i64> @llvm.aarch64.neon.vshld.n(<1 x i64> %vshl, i32 63)
  %0 = extractelement <1 x i64> %vshl1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vshld.n(<1 x i64>, i32)

define i64 @test_vshld_n_u64(i64 %a) {
; CHECK: test_vshld_n_u64
; CHECK: shl {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vshl = insertelement <1 x i64> undef, i64 %a, i32 0
  %vshl1 = call <1 x i64> @llvm.aarch64.neon.vshld.n(<1 x i64> %vshl, i32 63)
  %0 = extractelement <1 x i64> %vshl1, i32 0
  ret i64 %0
}

define i8 @test_vqshlb_n_s8(i8 %a) {
; CHECK: test_vqshlb_n_s8
; CHECK: sqshl {{b[0-9]+}}, {{b[0-9]+}}, #7
entry:
  %vsqshl = insertelement <1 x i8> undef, i8 %a, i32 0
  %vsqshl1 = call <1 x i8> @llvm.aarch64.neon.vqshls.n.v1i8(<1 x i8> %vsqshl, i32 7)
  %0 = extractelement <1 x i8> %vsqshl1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vqshls.n.v1i8(<1 x i8>, i32)

define i16 @test_vqshlh_n_s16(i16 %a) {
; CHECK: test_vqshlh_n_s16
; CHECK: sqshl {{h[0-9]+}}, {{h[0-9]+}}, #15
entry:
  %vsqshl = insertelement <1 x i16> undef, i16 %a, i32 0
  %vsqshl1 = call <1 x i16> @llvm.aarch64.neon.vqshls.n.v1i16(<1 x i16> %vsqshl, i32 15)
  %0 = extractelement <1 x i16> %vsqshl1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vqshls.n.v1i16(<1 x i16>, i32)

define i32 @test_vqshls_n_s32(i32 %a) {
; CHECK: test_vqshls_n_s32
; CHECK: sqshl {{s[0-9]+}}, {{s[0-9]+}}, #31
entry:
  %vsqshl = insertelement <1 x i32> undef, i32 %a, i32 0
  %vsqshl1 = call <1 x i32> @llvm.aarch64.neon.vqshls.n.v1i32(<1 x i32> %vsqshl, i32 31)
  %0 = extractelement <1 x i32> %vsqshl1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vqshls.n.v1i32(<1 x i32>, i32)

define i64 @test_vqshld_n_s64(i64 %a) {
; CHECK: test_vqshld_n_s64
; CHECK: sqshl {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsqshl = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsqshl1 = call <1 x i64> @llvm.aarch64.neon.vqshls.n.v1i64(<1 x i64> %vsqshl, i32 63)
  %0 = extractelement <1 x i64> %vsqshl1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vqshls.n.v1i64(<1 x i64>, i32)

define i8 @test_vqshlb_n_u8(i8 %a) {
; CHECK: test_vqshlb_n_u8
; CHECK: uqshl {{b[0-9]+}}, {{b[0-9]+}}, #7
entry:
  %vuqshl = insertelement <1 x i8> undef, i8 %a, i32 0
  %vuqshl1 = call <1 x i8> @llvm.aarch64.neon.vqshlu.n.v1i8(<1 x i8> %vuqshl, i32 7)
  %0 = extractelement <1 x i8> %vuqshl1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vqshlu.n.v1i8(<1 x i8>, i32)

define i16 @test_vqshlh_n_u16(i16 %a) {
; CHECK: test_vqshlh_n_u16
; CHECK: uqshl {{h[0-9]+}}, {{h[0-9]+}}, #15
entry:
  %vuqshl = insertelement <1 x i16> undef, i16 %a, i32 0
  %vuqshl1 = call <1 x i16> @llvm.aarch64.neon.vqshlu.n.v1i16(<1 x i16> %vuqshl, i32 15)
  %0 = extractelement <1 x i16> %vuqshl1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vqshlu.n.v1i16(<1 x i16>, i32)

define i32 @test_vqshls_n_u32(i32 %a) {
; CHECK: test_vqshls_n_u32
; CHECK: uqshl {{s[0-9]+}}, {{s[0-9]+}}, #31
entry:
  %vuqshl = insertelement <1 x i32> undef, i32 %a, i32 0
  %vuqshl1 = call <1 x i32> @llvm.aarch64.neon.vqshlu.n.v1i32(<1 x i32> %vuqshl, i32 31)
  %0 = extractelement <1 x i32> %vuqshl1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vqshlu.n.v1i32(<1 x i32>, i32)

define i64 @test_vqshld_n_u64(i64 %a) {
; CHECK: test_vqshld_n_u64
; CHECK: uqshl {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vuqshl = insertelement <1 x i64> undef, i64 %a, i32 0
  %vuqshl1 = call <1 x i64> @llvm.aarch64.neon.vqshlu.n.v1i64(<1 x i64> %vuqshl, i32 63)
  %0 = extractelement <1 x i64> %vuqshl1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vqshlu.n.v1i64(<1 x i64>, i32)

define i8 @test_vqshlub_n_s8(i8 %a) {
; CHECK: test_vqshlub_n_s8
; CHECK: sqshlu {{b[0-9]+}}, {{b[0-9]+}}, #7
entry:
  %vsqshlu = insertelement <1 x i8> undef, i8 %a, i32 0
  %vsqshlu1 = call <1 x i8> @llvm.aarch64.neon.vsqshlu.v1i8(<1 x i8> %vsqshlu, i32 7)
  %0 = extractelement <1 x i8> %vsqshlu1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vsqshlu.v1i8(<1 x i8>, i32)

define i16 @test_vqshluh_n_s16(i16 %a) {
; CHECK: test_vqshluh_n_s16
; CHECK: sqshlu {{h[0-9]+}}, {{h[0-9]+}}, #15
entry:
  %vsqshlu = insertelement <1 x i16> undef, i16 %a, i32 0
  %vsqshlu1 = call <1 x i16> @llvm.aarch64.neon.vsqshlu.v1i16(<1 x i16> %vsqshlu, i32 15)
  %0 = extractelement <1 x i16> %vsqshlu1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vsqshlu.v1i16(<1 x i16>, i32)

define i32 @test_vqshlus_n_s32(i32 %a) {
; CHECK: test_vqshlus_n_s32
; CHECK: sqshlu {{s[0-9]+}}, {{s[0-9]+}}, #31
entry:
  %vsqshlu = insertelement <1 x i32> undef, i32 %a, i32 0
  %vsqshlu1 = call <1 x i32> @llvm.aarch64.neon.vsqshlu.v1i32(<1 x i32> %vsqshlu, i32 31)
  %0 = extractelement <1 x i32> %vsqshlu1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vsqshlu.v1i32(<1 x i32>, i32)

define i64 @test_vqshlud_n_s64(i64 %a) {
; CHECK: test_vqshlud_n_s64
; CHECK: sqshlu {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsqshlu = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsqshlu1 = call <1 x i64> @llvm.aarch64.neon.vsqshlu.v1i64(<1 x i64> %vsqshlu, i32 63)
  %0 = extractelement <1 x i64> %vsqshlu1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vsqshlu.v1i64(<1 x i64>, i32)

define i64 @test_vsrid_n_s64(i64 %a, i64 %b) {
; CHECK: test_vsrid_n_s64
; CHECK: sri {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsri = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsri1 = insertelement <1 x i64> undef, i64 %b, i32 0
  %vsri2 = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> %vsri, <1 x i64> %vsri1, i32 63)
  %0 = extractelement <1 x i64> %vsri2, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64>, <1 x i64>, i32)

define i64 @test_vsrid_n_u64(i64 %a, i64 %b) {
; CHECK: test_vsrid_n_u64
; CHECK: sri {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsri = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsri1 = insertelement <1 x i64> undef, i64 %b, i32 0
  %vsri2 = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> %vsri, <1 x i64> %vsri1, i32 63)
  %0 = extractelement <1 x i64> %vsri2, i32 0
  ret i64 %0
}

define i64 @test_vslid_n_s64(i64 %a, i64 %b) {
; CHECK: test_vslid_n_s64
; CHECK: sli {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsli = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsli1 = insertelement <1 x i64> undef, i64 %b, i32 0
  %vsli2 = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> %vsli, <1 x i64> %vsli1, i32 63)
  %0 = extractelement <1 x i64> %vsli2, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64>, <1 x i64>, i32)

define i64 @test_vslid_n_u64(i64 %a, i64 %b) {
; CHECK: test_vslid_n_u64
; CHECK: sli {{d[0-9]+}}, {{d[0-9]+}}, #63
entry:
  %vsli = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsli1 = insertelement <1 x i64> undef, i64 %b, i32 0
  %vsli2 = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> %vsli, <1 x i64> %vsli1, i32 63)
  %0 = extractelement <1 x i64> %vsli2, i32 0
  ret i64 %0
}

define i8 @test_vqshrnh_n_s16(i16 %a) {
; CHECK: test_vqshrnh_n_s16
; CHECK: sqshrn {{b[0-9]+}}, {{h[0-9]+}}, #8
entry:
  %vsqshrn = insertelement <1 x i16> undef, i16 %a, i32 0
  %vsqshrn1 = call <1 x i8> @llvm.aarch64.neon.vsqshrn.v1i8(<1 x i16> %vsqshrn, i32 8)
  %0 = extractelement <1 x i8> %vsqshrn1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vsqshrn.v1i8(<1 x i16>, i32)

define i16 @test_vqshrns_n_s32(i32 %a) {
; CHECK: test_vqshrns_n_s32
; CHECK: sqshrn {{h[0-9]+}}, {{s[0-9]+}}, #16
entry:
  %vsqshrn = insertelement <1 x i32> undef, i32 %a, i32 0
  %vsqshrn1 = call <1 x i16> @llvm.aarch64.neon.vsqshrn.v1i16(<1 x i32> %vsqshrn, i32 16)
  %0 = extractelement <1 x i16> %vsqshrn1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vsqshrn.v1i16(<1 x i32>, i32)

define i32 @test_vqshrnd_n_s64(i64 %a) {
; CHECK: test_vqshrnd_n_s64
; CHECK: sqshrn {{s[0-9]+}}, {{d[0-9]+}}, #32
entry:
  %vsqshrn = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsqshrn1 = call <1 x i32> @llvm.aarch64.neon.vsqshrn.v1i32(<1 x i64> %vsqshrn, i32 32)
  %0 = extractelement <1 x i32> %vsqshrn1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vsqshrn.v1i32(<1 x i64>, i32)

define i8 @test_vqshrnh_n_u16(i16 %a) {
; CHECK: test_vqshrnh_n_u16
; CHECK: uqshrn {{b[0-9]+}}, {{h[0-9]+}}, #8
entry:
  %vuqshrn = insertelement <1 x i16> undef, i16 %a, i32 0
  %vuqshrn1 = call <1 x i8> @llvm.aarch64.neon.vuqshrn.v1i8(<1 x i16> %vuqshrn, i32 8)
  %0 = extractelement <1 x i8> %vuqshrn1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vuqshrn.v1i8(<1 x i16>, i32)

define i16 @test_vqshrns_n_u32(i32 %a) {
; CHECK: test_vqshrns_n_u32
; CHECK: uqshrn {{h[0-9]+}}, {{s[0-9]+}}, #16
entry:
  %vuqshrn = insertelement <1 x i32> undef, i32 %a, i32 0
  %vuqshrn1 = call <1 x i16> @llvm.aarch64.neon.vuqshrn.v1i16(<1 x i32> %vuqshrn, i32 16)
  %0 = extractelement <1 x i16> %vuqshrn1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vuqshrn.v1i16(<1 x i32>, i32)

define i32 @test_vqshrnd_n_u64(i64 %a) {
; CHECK: test_vqshrnd_n_u64
; CHECK: uqshrn {{s[0-9]+}}, {{d[0-9]+}}, #32
entry:
  %vuqshrn = insertelement <1 x i64> undef, i64 %a, i32 0
  %vuqshrn1 = call <1 x i32> @llvm.aarch64.neon.vuqshrn.v1i32(<1 x i64> %vuqshrn, i32 32)
  %0 = extractelement <1 x i32> %vuqshrn1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vuqshrn.v1i32(<1 x i64>, i32)

define i8 @test_vqrshrnh_n_s16(i16 %a) {
; CHECK: test_vqrshrnh_n_s16
; CHECK: sqrshrn {{b[0-9]+}}, {{h[0-9]+}}, #8
entry:
  %vsqrshrn = insertelement <1 x i16> undef, i16 %a, i32 0
  %vsqrshrn1 = call <1 x i8> @llvm.aarch64.neon.vsqrshrn.v1i8(<1 x i16> %vsqrshrn, i32 8)
  %0 = extractelement <1 x i8> %vsqrshrn1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vsqrshrn.v1i8(<1 x i16>, i32)

define i16 @test_vqrshrns_n_s32(i32 %a) {
; CHECK: test_vqrshrns_n_s32
; CHECK: sqrshrn {{h[0-9]+}}, {{s[0-9]+}}, #16
entry:
  %vsqrshrn = insertelement <1 x i32> undef, i32 %a, i32 0
  %vsqrshrn1 = call <1 x i16> @llvm.aarch64.neon.vsqrshrn.v1i16(<1 x i32> %vsqrshrn, i32 16)
  %0 = extractelement <1 x i16> %vsqrshrn1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vsqrshrn.v1i16(<1 x i32>, i32)

define i32 @test_vqrshrnd_n_s64(i64 %a) {
; CHECK: test_vqrshrnd_n_s64
; CHECK: sqrshrn {{s[0-9]+}}, {{d[0-9]+}}, #32
entry:
  %vsqrshrn = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsqrshrn1 = call <1 x i32> @llvm.aarch64.neon.vsqrshrn.v1i32(<1 x i64> %vsqrshrn, i32 32)
  %0 = extractelement <1 x i32> %vsqrshrn1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vsqrshrn.v1i32(<1 x i64>, i32)

define i8 @test_vqrshrnh_n_u16(i16 %a) {
; CHECK: test_vqrshrnh_n_u16
; CHECK: uqrshrn {{b[0-9]+}}, {{h[0-9]+}}, #8
entry:
  %vuqrshrn = insertelement <1 x i16> undef, i16 %a, i32 0
  %vuqrshrn1 = call <1 x i8> @llvm.aarch64.neon.vuqrshrn.v1i8(<1 x i16> %vuqrshrn, i32 8)
  %0 = extractelement <1 x i8> %vuqrshrn1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vuqrshrn.v1i8(<1 x i16>, i32)

define i16 @test_vqrshrns_n_u32(i32 %a) {
; CHECK: test_vqrshrns_n_u32
; CHECK: uqrshrn {{h[0-9]+}}, {{s[0-9]+}}, #16
entry:
  %vuqrshrn = insertelement <1 x i32> undef, i32 %a, i32 0
  %vuqrshrn1 = call <1 x i16> @llvm.aarch64.neon.vuqrshrn.v1i16(<1 x i32> %vuqrshrn, i32 16)
  %0 = extractelement <1 x i16> %vuqrshrn1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vuqrshrn.v1i16(<1 x i32>, i32)

define i32 @test_vqrshrnd_n_u64(i64 %a) {
; CHECK: test_vqrshrnd_n_u64
; CHECK: uqrshrn {{s[0-9]+}}, {{d[0-9]+}}, #32
entry:
  %vuqrshrn = insertelement <1 x i64> undef, i64 %a, i32 0
  %vuqrshrn1 = call <1 x i32> @llvm.aarch64.neon.vuqrshrn.v1i32(<1 x i64> %vuqrshrn, i32 32)
  %0 = extractelement <1 x i32> %vuqrshrn1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vuqrshrn.v1i32(<1 x i64>, i32)

define i8 @test_vqshrunh_n_s16(i16 %a) {
; CHECK: test_vqshrunh_n_s16
; CHECK: sqshrun {{b[0-9]+}}, {{h[0-9]+}}, #8
entry:
  %vsqshrun = insertelement <1 x i16> undef, i16 %a, i32 0
  %vsqshrun1 = call <1 x i8> @llvm.aarch64.neon.vsqshrun.v1i8(<1 x i16> %vsqshrun, i32 8)
  %0 = extractelement <1 x i8> %vsqshrun1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vsqshrun.v1i8(<1 x i16>, i32)

define i16 @test_vqshruns_n_s32(i32 %a) {
; CHECK: test_vqshruns_n_s32
; CHECK: sqshrun {{h[0-9]+}}, {{s[0-9]+}}, #16
entry:
  %vsqshrun = insertelement <1 x i32> undef, i32 %a, i32 0
  %vsqshrun1 = call <1 x i16> @llvm.aarch64.neon.vsqshrun.v1i16(<1 x i32> %vsqshrun, i32 16)
  %0 = extractelement <1 x i16> %vsqshrun1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vsqshrun.v1i16(<1 x i32>, i32)

define i32 @test_vqshrund_n_s64(i64 %a) {
; CHECK: test_vqshrund_n_s64
; CHECK: sqshrun {{s[0-9]+}}, {{d[0-9]+}}, #32
entry:
  %vsqshrun = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsqshrun1 = call <1 x i32> @llvm.aarch64.neon.vsqshrun.v1i32(<1 x i64> %vsqshrun, i32 32)
  %0 = extractelement <1 x i32> %vsqshrun1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vsqshrun.v1i32(<1 x i64>, i32)

define i8 @test_vqrshrunh_n_s16(i16 %a) {
; CHECK: test_vqrshrunh_n_s16
; CHECK: sqrshrun {{b[0-9]+}}, {{h[0-9]+}}, #8
entry:
  %vsqrshrun = insertelement <1 x i16> undef, i16 %a, i32 0
  %vsqrshrun1 = call <1 x i8> @llvm.aarch64.neon.vsqrshrun.v1i8(<1 x i16> %vsqrshrun, i32 8)
  %0 = extractelement <1 x i8> %vsqrshrun1, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vsqrshrun.v1i8(<1 x i16>, i32)

define i16 @test_vqrshruns_n_s32(i32 %a) {
; CHECK: test_vqrshruns_n_s32
; CHECK: sqrshrun {{h[0-9]+}}, {{s[0-9]+}}, #16
entry:
  %vsqrshrun = insertelement <1 x i32> undef, i32 %a, i32 0
  %vsqrshrun1 = call <1 x i16> @llvm.aarch64.neon.vsqrshrun.v1i16(<1 x i32> %vsqrshrun, i32 16)
  %0 = extractelement <1 x i16> %vsqrshrun1, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vsqrshrun.v1i16(<1 x i32>, i32)

define i32 @test_vqrshrund_n_s64(i64 %a) {
; CHECK: test_vqrshrund_n_s64
; CHECK: sqrshrun {{s[0-9]+}}, {{d[0-9]+}}, #32
entry:
  %vsqrshrun = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsqrshrun1 = call <1 x i32> @llvm.aarch64.neon.vsqrshrun.v1i32(<1 x i64> %vsqrshrun, i32 32)
  %0 = extractelement <1 x i32> %vsqrshrun1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vsqrshrun.v1i32(<1 x i64>, i32)
