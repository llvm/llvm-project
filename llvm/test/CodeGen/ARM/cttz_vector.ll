; RUN: llc < %s -mtriple armv7-linux-gnueabihf -mattr=+neon | FileCheck %s

; This test checks the @llvm.cttz.* intrinsics for vectors.

declare <1 x i8> @llvm.cttz.v1i8(<1 x i8>, i1)
declare <2 x i8> @llvm.cttz.v2i8(<2 x i8>, i1)
declare <4 x i8> @llvm.cttz.v4i8(<4 x i8>, i1)
declare <8 x i8> @llvm.cttz.v8i8(<8 x i8>, i1)
declare <16 x i8> @llvm.cttz.v16i8(<16 x i8>, i1)

declare <1 x i16> @llvm.cttz.v1i16(<1 x i16>, i1)
declare <2 x i16> @llvm.cttz.v2i16(<2 x i16>, i1)
declare <4 x i16> @llvm.cttz.v4i16(<4 x i16>, i1)
declare <8 x i16> @llvm.cttz.v8i16(<8 x i16>, i1)

declare <1 x i32> @llvm.cttz.v1i32(<1 x i32>, i1)
declare <2 x i32> @llvm.cttz.v2i32(<2 x i32>, i1)
declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1)

declare <1 x i64> @llvm.cttz.v1i64(<1 x i64>, i1)
declare <2 x i64> @llvm.cttz.v2i64(<2 x i64>, i1)

;------------------------------------------------------------------------------

define void @test_v1i8(<1 x i8>* %p) {
; CHECK-LABEL: test_v1i8
  %a = load <1 x i8>, <1 x i8>* %p
  %tmp = call <1 x i8> @llvm.cttz.v1i8(<1 x i8> %a, i1 false)
  store <1 x i8> %tmp, <1 x i8>* %p
  ret void
}

define void @test_v2i8(<2 x i8>* %p) {
; CHECK-LABEL: test_v2i8:
  %a = load <2 x i8>, <2 x i8>* %p
  %tmp = call <2 x i8> @llvm.cttz.v2i8(<2 x i8> %a, i1 false)
  store <2 x i8> %tmp, <2 x i8>* %p
  ret void
}

define void @test_v4i8(<4 x i8>* %p) {
; CHECK-LABEL: test_v4i8:
  %a = load <4 x i8>, <4 x i8>* %p
  %tmp = call <4 x i8> @llvm.cttz.v4i8(<4 x i8> %a, i1 false)
  store <4 x i8> %tmp, <4 x i8>* %p
  ret void
}

define void @test_v8i8(<8 x i8>* %p) {
; CHECK-LABEL: test_v8i8:
; CHECK: vldr		[[D1:d[0-9]+]], [r0]
; CHECK: vmov.i8	[[D2:d[0-9]+]], #0x1
; CHECK: vneg.s8	[[D3:d[0-9]+]], [[D1]]
; CHECK: vand		[[D1]], [[D1]], [[D3]]
; CHECK: vsub.i8	[[D1]], [[D1]], [[D2]]
; CHECK: vcnt.8		[[D1]], [[D1]]
; CHECK: vstr		[[D1]], [r0]
  %a = load <8 x i8>, <8 x i8>* %p
  %tmp = call <8 x i8> @llvm.cttz.v8i8(<8 x i8> %a, i1 false)
  store <8 x i8> %tmp, <8 x i8>* %p
  ret void
}

define void @test_v16i8(<16 x i8>* %p) {
; CHECK-LABEL: test_v16i8:
; CHECK: vld1.64	{[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [r0]
; CHECK: vmov.i8	[[Q2:q[0-9]+]], #0x1
; CHECK: vneg.s8	[[Q3:q[0-9]+]], [[Q1:q[0-9]+]]
; CHECK: vand		[[Q1]], [[Q1]], [[Q3]]
; CHECK: vsub.i8	[[Q1]], [[Q1]], [[Q2]]
; CHECK: vcnt.8		[[Q1]], [[Q1]]
; CHECK: vst1.64	{[[D1]], [[D2]]}, [r0]
  %a = load <16 x i8>, <16 x i8>* %p
  %tmp = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 false)
  store <16 x i8> %tmp, <16 x i8>* %p
  ret void
}

define void @test_v1i16(<1 x i16>* %p) {
; CHECK-LABEL: test_v1i16:
  %a = load <1 x i16>, <1 x i16>* %p
  %tmp = call <1 x i16> @llvm.cttz.v1i16(<1 x i16> %a, i1 false)
  store <1 x i16> %tmp, <1 x i16>* %p
  ret void
}

define void @test_v2i16(<2 x i16>* %p) {
; CHECK-LABEL: test_v2i16:
  %a = load <2 x i16>, <2 x i16>* %p
  %tmp = call <2 x i16> @llvm.cttz.v2i16(<2 x i16> %a, i1 false)
  store <2 x i16> %tmp, <2 x i16>* %p
  ret void
}

define void @test_v4i16(<4 x i16>* %p) {
; CHECK-LABEL: test_v4i16:
; CHECK: vldr		[[D1:d[0-9]+]], [r0]
; CHECK: vmov.i16	[[D2:d[0-9]+]], #0x1
; CHECK: vneg.s16	[[D3:d[0-9]+]], [[D1]]
; CHECK: vand		[[D1]], [[D1]], [[D3]]
; CHECK: vsub.i16	[[D1]], [[D1]], [[D2]]
; CHECK: vcnt.8		[[D1]], [[D1]]
; CHECK: vpaddl.u8	[[D1]], [[D1]]
; CHECK: vstr		[[D1]], [r0]
  %a = load <4 x i16>, <4 x i16>* %p
  %tmp = call <4 x i16> @llvm.cttz.v4i16(<4 x i16> %a, i1 false)
  store <4 x i16> %tmp, <4 x i16>* %p
  ret void
}

define void @test_v8i16(<8 x i16>* %p) {
; CHECK-LABEL: test_v8i16:
; CHECK: vld1.64	{[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [r0]
; CHECK: vmov.i16	[[Q2:q[0-9]+]], #0x1
; CHECK: vneg.s16	[[Q3:q[0-9]+]], [[Q1:q[0-9]+]]
; CHECK: vand		[[Q1]], [[Q1]], [[Q3]]
; CHECK: vsub.i16	[[Q1]], [[Q1]], [[Q2]]
; CHECK: vcnt.8		[[Q1]], [[Q1]]
; CHECK: vpaddl.u8	[[Q1]], [[Q1]]
; CHECK: vst1.64	{[[D1]], [[D2]]}, [r0]
  %a = load <8 x i16>, <8 x i16>* %p
  %tmp = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 false)
  store <8 x i16> %tmp, <8 x i16>* %p
  ret void
}

define void @test_v1i32(<1 x i32>* %p) {
; CHECK-LABEL: test_v1i32:
  %a = load <1 x i32>, <1 x i32>* %p
  %tmp = call <1 x i32> @llvm.cttz.v1i32(<1 x i32> %a, i1 false)
  store <1 x i32> %tmp, <1 x i32>* %p
  ret void
}

define void @test_v2i32(<2 x i32>* %p) {
; CHECK-LABEL: test_v2i32:
; CHECK: vldr		[[D1:d[0-9]+]], [r0]
; CHECK: vmov.i32	[[D2:d[0-9]+]], #0x1
; CHECK: vneg.s32	[[D3:d[0-9]+]], [[D1]]
; CHECK: vand		[[D1]], [[D1]], [[D3]]
; CHECK: vsub.i32	[[D1]], [[D1]], [[D2]]
; CHECK: vcnt.8		[[D1]], [[D1]]
; CHECK: vpaddl.u8	[[D1]], [[D1]]
; CHECK: vpaddl.u16	[[D1]], [[D1]]
; CHECK: vstr		[[D1]], [r0]
  %a = load <2 x i32>, <2 x i32>* %p
  %tmp = call <2 x i32> @llvm.cttz.v2i32(<2 x i32> %a, i1 false)
  store <2 x i32> %tmp, <2 x i32>* %p
  ret void
}

define void @test_v4i32(<4 x i32>* %p) {
; CHECK-LABEL: test_v4i32:
; CHECK: vld1.64	{[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [r0]
; CHECK: vmov.i32	[[Q2:q[0-9]+]], #0x1
; CHECK: vneg.s32	[[Q3:q[0-9]+]], [[Q1:q[0-9]+]]
; CHECK: vand		[[Q1]], [[Q1]], [[Q3]]
; CHECK: vsub.i32	[[Q1]], [[Q1]], [[Q2]]
; CHECK: vcnt.8		[[Q1]], [[Q1]]
; CHECK: vpaddl.u8	[[Q1]], [[Q1]]
; CHECK: vpaddl.u16	[[Q1]], [[Q1]]
; CHECK: vst1.64	{[[D1]], [[D2]]}, [r0]
  %a = load <4 x i32>, <4 x i32>* %p
  %tmp = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 false)
  store <4 x i32> %tmp, <4 x i32>* %p
  ret void
}

define void @test_v1i64(<1 x i64>* %p) {
; CHECK-LABEL: test_v1i64:
; CHECK: vmov.i32	[[D2:d[0-9]+]], #0x0
; CHECK: vldr		[[D1:d[0-9]+]], [r0]
; CHECK: vmov.i64	[[D3:d[0-9]+]], #0xffffffffffffffff
; CHECK: vsub.i64	[[D2]], [[D2]], [[D1]]
; CHECK: vand		[[D2]], [[D1]], [[D2]]
; CHECK: vadd.i64	[[D2]], [[D2]], [[D3]]
; CHECK: vcnt.8		[[D2]], [[D2]]
; CHECK: vpaddl.u8	[[D2]], [[D2]]
; CHECK: vpaddl.u16	[[D2]], [[D2]]
; CHECK: vpaddl.u32	[[D2]], [[D2]]
; CHECK: vstr		[[D2]], [r0]
  %a = load <1 x i64>, <1 x i64>* %p
  %tmp = call <1 x i64> @llvm.cttz.v1i64(<1 x i64> %a, i1 false)
  store <1 x i64> %tmp, <1 x i64>* %p
  ret void
}

define void @test_v2i64(<2 x i64>* %p) {
; CHECK-LABEL: test_v2i64:
; CHECK: vmov.i32	[[Q2:q[0-9]+]], #0x0
; CHECK: vld1.64	{[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [r0]
; CHECK: vmov.i64	[[Q3:q[0-9]+]], #0xffffffffffffffff
; CHECK: vsub.i64	[[Q2]], [[Q2]], [[Q1:q[0-9]+]]
; CHECK: vand		[[Q2]], [[Q1]], [[Q2]]
; CHECK: vadd.i64	[[Q2]], [[Q2]], [[Q3]]
; CHECK: vcnt.8		[[Q2]], [[Q2]]
; CHECK: vpaddl.u8	[[Q2]], [[Q2]]
; CHECK: vpaddl.u16	[[Q2]], [[Q2]]
; CHECK: vpaddl.u32	[[Q2]], [[Q2]]
; CHECK: vst1.64	{d{{[0-9]+}}, d{{[0-9]+}}}, [r0]
  %a = load <2 x i64>, <2 x i64>* %p
  %tmp = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 false)
  store <2 x i64> %tmp, <2 x i64>* %p
  ret void
}

;------------------------------------------------------------------------------

define void @test_v1i8_zero_undef(<1 x i8>* %p) {
; CHECK-LABEL: test_v1i8_zero_undef
  %a = load <1 x i8>, <1 x i8>* %p
  %tmp = call <1 x i8> @llvm.cttz.v1i8(<1 x i8> %a, i1 true)
  store <1 x i8> %tmp, <1 x i8>* %p
  ret void
}

define void @test_v2i8_zero_undef(<2 x i8>* %p) {
; CHECK-LABEL: test_v2i8_zero_undef:
  %a = load <2 x i8>, <2 x i8>* %p
  %tmp = call <2 x i8> @llvm.cttz.v2i8(<2 x i8> %a, i1 true)
  store <2 x i8> %tmp, <2 x i8>* %p
  ret void
}

define void @test_v4i8_zero_undef(<4 x i8>* %p) {
; CHECK-LABEL: test_v4i8_zero_undef:
  %a = load <4 x i8>, <4 x i8>* %p
  %tmp = call <4 x i8> @llvm.cttz.v4i8(<4 x i8> %a, i1 true)
  store <4 x i8> %tmp, <4 x i8>* %p
  ret void
}

define void @test_v8i8_zero_undef(<8 x i8>* %p) {
; CHECK-LABEL: test_v8i8_zero_undef:
; CHECK: vldr		[[D1:d[0-9]+]], [r0]
; CHECK: vmov.i8	[[D2:d[0-9]+]], #0x1
; CHECK: vneg.s8	[[D3:d[0-9]+]], [[D1]]
; CHECK: vand		[[D1]], [[D1]], [[D3]]
; CHECK: vsub.i8	[[D1]], [[D1]], [[D2]]
; CHECK: vcnt.8		[[D1]], [[D1]]
; CHECK: vstr		[[D1]], [r0]
  %a = load <8 x i8>, <8 x i8>* %p
  %tmp = call <8 x i8> @llvm.cttz.v8i8(<8 x i8> %a, i1 true)
  store <8 x i8> %tmp, <8 x i8>* %p
  ret void
}

define void @test_v16i8_zero_undef(<16 x i8>* %p) {
; CHECK-LABEL: test_v16i8_zero_undef:
; CHECK: vld1.64	{[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [r0]
; CHECK: vmov.i8	[[Q2:q[0-9]+]], #0x1
; CHECK: vneg.s8	[[Q3:q[0-9]+]], [[Q1:q[0-9]+]]
; CHECK: vand		[[Q1]], [[Q1]], [[Q3]]
; CHECK: vsub.i8	[[Q1]], [[Q1]], [[Q2]]
; CHECK: vcnt.8		[[Q1]], [[Q1]]
; CHECK: vst1.64	{[[D1]], [[D2]]}, [r0]
  %a = load <16 x i8>, <16 x i8>* %p
  %tmp = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 true)
  store <16 x i8> %tmp, <16 x i8>* %p
  ret void
}

define void @test_v1i16_zero_undef(<1 x i16>* %p) {
; CHECK-LABEL: test_v1i16_zero_undef:
  %a = load <1 x i16>, <1 x i16>* %p
  %tmp = call <1 x i16> @llvm.cttz.v1i16(<1 x i16> %a, i1 true)
  store <1 x i16> %tmp, <1 x i16>* %p
  ret void
}

define void @test_v2i16_zero_undef(<2 x i16>* %p) {
; CHECK-LABEL: test_v2i16_zero_undef:
  %a = load <2 x i16>, <2 x i16>* %p
  %tmp = call <2 x i16> @llvm.cttz.v2i16(<2 x i16> %a, i1 true)
  store <2 x i16> %tmp, <2 x i16>* %p
  ret void
}

define void @test_v4i16_zero_undef(<4 x i16>* %p) {
; CHECK-LABEL: test_v4i16_zero_undef:
; CHECK: vldr		[[D1:d[0-9]+]], [r0]
; CHECK: vneg.s16	[[D2:d[0-9]+]], [[D1]]
; CHECK: vand		[[D1]], [[D1]], [[D2]]
; CHECK: vmov.i16	[[D3:d[0-9]+]], #0xf
; CHECK: vclz.i16	[[D1]], [[D1]]
; CHECK: vsub.i16	[[D1]], [[D3]], [[D1]]
; CHECK: vstr		[[D1]], [r0]
  %a = load <4 x i16>, <4 x i16>* %p
  %tmp = call <4 x i16> @llvm.cttz.v4i16(<4 x i16> %a, i1 true)
  store <4 x i16> %tmp, <4 x i16>* %p
  ret void
}

define void @test_v8i16_zero_undef(<8 x i16>* %p) {
; CHECK-LABEL: test_v8i16_zero_undef:
; CHECK: vld1.64	{[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [r0]
; CHECK: vneg.s16	[[Q2:q[0-9]+]], [[Q1:q[0-9]+]]
; CHECK: vand		[[Q1]], [[Q1]], [[Q2]]
; CHECK: vmov.i16	[[Q3:q[0-9]+]], #0xf
; CHECK: vclz.i16	[[Q1]], [[Q1]]
; CHECK: vsub.i16	[[Q1]], [[Q3]], [[Q1]]
; CHECK: vst1.64	{[[D1]], [[D2]]}, [r0]
  %a = load <8 x i16>, <8 x i16>* %p
  %tmp = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 true)
  store <8 x i16> %tmp, <8 x i16>* %p
  ret void
}

define void @test_v1i32_zero_undef(<1 x i32>* %p) {
; CHECK-LABEL: test_v1i32_zero_undef:
  %a = load <1 x i32>, <1 x i32>* %p
  %tmp = call <1 x i32> @llvm.cttz.v1i32(<1 x i32> %a, i1 true)
  store <1 x i32> %tmp, <1 x i32>* %p
  ret void
}

define void @test_v2i32_zero_undef(<2 x i32>* %p) {
; CHECK-LABEL: test_v2i32_zero_undef:
; CHECK: vldr		[[D1:d[0-9]+]], [r0]
; CHECK: vneg.s32	[[D2:d[0-9]+]], [[D1]]
; CHECK: vand		[[D1]], [[D1]], [[D2]]
; CHECK: vmov.i32	[[D3:d[0-9]+]], #0x1f
; CHECK: vclz.i32	[[D1]], [[D1]]
; CHECK: vsub.i32	[[D1]], [[D3]], [[D1]]
; CHECK: vstr		[[D1]], [r0]
  %a = load <2 x i32>, <2 x i32>* %p
  %tmp = call <2 x i32> @llvm.cttz.v2i32(<2 x i32> %a, i1 true)
  store <2 x i32> %tmp, <2 x i32>* %p
  ret void
}

define void @test_v4i32_zero_undef(<4 x i32>* %p) {
; CHECK-LABEL: test_v4i32_zero_undef:
; CHECK: vld1.64	{[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [r0]
; CHECK: vneg.s32	[[Q2:q[0-9]+]], [[Q1:q[0-9]+]]
; CHECK: vand		[[Q1]], [[Q1]], [[Q2]]
; CHECK: vmov.i32	[[Q3:q[0-9]+]], #0x1f
; CHECK: vclz.i32	[[Q1]], [[Q1]]
; CHECK: vsub.i32	[[Q1]], [[Q3]], [[Q1]]
; CHECK: vst1.64	{[[D1]], [[D2]]}, [r0]
  %a = load <4 x i32>, <4 x i32>* %p
  %tmp = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 true)
  store <4 x i32> %tmp, <4 x i32>* %p
  ret void
}

define void @test_v1i64_zero_undef(<1 x i64>* %p) {
; CHECK-LABEL: test_v1i64_zero_undef:
; CHECK: vmov.i32	[[D2:d[0-9]+]], #0x0
; CHECK: vldr		[[D1:d[0-9]+]], [r0]
; CHECK: vmov.i64	[[D3:d[0-9]+]], #0xffffffffffffffff
; CHECK: vsub.i64	[[D2]], [[D2]], [[D1]]
; CHECK: vand		[[D2]], [[D1]], [[D2]]
; CHECK: vadd.i64	[[D2]], [[D2]], [[D3]]
; CHECK: vcnt.8		[[D2]], [[D2]]
; CHECK: vpaddl.u8	[[D2]], [[D2]]
; CHECK: vpaddl.u16	[[D2]], [[D2]]
; CHECK: vpaddl.u32	[[D2]], [[D2]]
; CHECK: vstr		[[D2]], [r0]
  %a = load <1 x i64>, <1 x i64>* %p
  %tmp = call <1 x i64> @llvm.cttz.v1i64(<1 x i64> %a, i1 true)
  store <1 x i64> %tmp, <1 x i64>* %p
  ret void
}

define void @test_v2i64_zero_undef(<2 x i64>* %p) {
; CHECK-LABEL: test_v2i64_zero_undef:
; CHECK: vmov.i32	[[Q2:q[0-9]+]], #0x0
; CHECK: vld1.64	{[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [r0]
; CHECK: vmov.i64	[[Q3:q[0-9]+]], #0xffffffffffffffff
; CHECK: vsub.i64	[[Q2]], [[Q2]], [[Q1:q[0-9]+]]
; CHECK: vand		[[Q2]], [[Q1]], [[Q2]]
; CHECK: vadd.i64	[[Q2]], [[Q2]], [[Q3]]
; CHECK: vcnt.8		[[Q2]], [[Q2]]
; CHECK: vpaddl.u8	[[Q2]], [[Q2]]
; CHECK: vpaddl.u16	[[Q2]], [[Q2]]
; CHECK: vpaddl.u32	[[Q2]], [[Q2]]
; CHECK: vst1.64	{d{{[0-9]+}}, d{{[0-9]+}}}, [r0]
  %a = load <2 x i64>, <2 x i64>* %p
  %tmp = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 true)
  store <2 x i64> %tmp, <2 x i64>* %p
  ret void
}
