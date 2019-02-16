; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -verify-machineinstrs -asm-verbose=false | FileCheck %s

%struct.__neon_int8x8x2_t = type { <8 x i8>,  <8 x i8> }
%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int8x8x4_t = type { <8 x i8>,  <8 x i8>, <8 x i8>,  <8 x i8> }

define %struct.__neon_int8x8x2_t @ld2_8b(i8* %A) nounwind {
; CHECK-LABEL: ld2_8b
; Make sure we are loading into the results defined by the ABI (i.e., v0, v1)
; and from the argument of the function also defined by ABI (i.e., x0)
; CHECK: ld2.8b { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x8x2_t @llvm.aarch64.neon.ld2.v8i8.p0i8(i8* %A)
	ret %struct.__neon_int8x8x2_t  %tmp2
}

define %struct.__neon_int8x8x3_t @ld3_8b(i8* %A) nounwind {
; CHECK-LABEL: ld3_8b
; Make sure we are using the operands defined by the ABI
; CHECK: ld3.8b { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x8x3_t @llvm.aarch64.neon.ld3.v8i8.p0i8(i8* %A)
	ret %struct.__neon_int8x8x3_t  %tmp2
}

define %struct.__neon_int8x8x4_t @ld4_8b(i8* %A) nounwind {
; CHECK-LABEL: ld4_8b
; Make sure we are using the operands defined by the ABI
; CHECK: ld4.8b { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x8x4_t @llvm.aarch64.neon.ld4.v8i8.p0i8(i8* %A)
	ret %struct.__neon_int8x8x4_t  %tmp2
}

declare %struct.__neon_int8x8x2_t @llvm.aarch64.neon.ld2.v8i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int8x8x3_t @llvm.aarch64.neon.ld3.v8i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int8x8x4_t @llvm.aarch64.neon.ld4.v8i8.p0i8(i8*) nounwind readonly

%struct.__neon_int8x16x2_t = type { <16 x i8>,  <16 x i8> }
%struct.__neon_int8x16x3_t = type { <16 x i8>,  <16 x i8>,  <16 x i8> }
%struct.__neon_int8x16x4_t = type { <16 x i8>,  <16 x i8>, <16 x i8>,  <16 x i8> }

define %struct.__neon_int8x16x2_t @ld2_16b(i8* %A) nounwind {
; CHECK-LABEL: ld2_16b
; Make sure we are using the operands defined by the ABI
; CHECK: ld2.16b { v0, v1 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int8x16x2_t @llvm.aarch64.neon.ld2.v16i8.p0i8(i8* %A)
  ret %struct.__neon_int8x16x2_t  %tmp2
}

define %struct.__neon_int8x16x3_t @ld3_16b(i8* %A) nounwind {
; CHECK-LABEL: ld3_16b
; Make sure we are using the operands defined by the ABI
; CHECK: ld3.16b { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int8x16x3_t @llvm.aarch64.neon.ld3.v16i8.p0i8(i8* %A)
  ret %struct.__neon_int8x16x3_t  %tmp2
}

define %struct.__neon_int8x16x4_t @ld4_16b(i8* %A) nounwind {
; CHECK-LABEL: ld4_16b
; Make sure we are using the operands defined by the ABI
; CHECK: ld4.16b { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int8x16x4_t @llvm.aarch64.neon.ld4.v16i8.p0i8(i8* %A)
  ret %struct.__neon_int8x16x4_t  %tmp2
}

declare %struct.__neon_int8x16x2_t @llvm.aarch64.neon.ld2.v16i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int8x16x3_t @llvm.aarch64.neon.ld3.v16i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int8x16x4_t @llvm.aarch64.neon.ld4.v16i8.p0i8(i8*) nounwind readonly

%struct.__neon_int16x4x2_t = type { <4 x i16>,  <4 x i16> }
%struct.__neon_int16x4x3_t = type { <4 x i16>,  <4 x i16>,  <4 x i16> }
%struct.__neon_int16x4x4_t = type { <4 x i16>,  <4 x i16>, <4 x i16>,  <4 x i16> }

define %struct.__neon_int16x4x2_t @ld2_4h(i16* %A) nounwind {
; CHECK-LABEL: ld2_4h
; Make sure we are using the operands defined by the ABI
; CHECK: ld2.4h { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x4x2_t @llvm.aarch64.neon.ld2.v4i16.p0i16(i16* %A)
	ret %struct.__neon_int16x4x2_t  %tmp2
}

define %struct.__neon_int16x4x3_t @ld3_4h(i16* %A) nounwind {
; CHECK-LABEL: ld3_4h
; Make sure we are using the operands defined by the ABI
; CHECK: ld3.4h { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x4x3_t @llvm.aarch64.neon.ld3.v4i16.p0i16(i16* %A)
	ret %struct.__neon_int16x4x3_t  %tmp2
}

define %struct.__neon_int16x4x4_t @ld4_4h(i16* %A) nounwind {
; CHECK-LABEL: ld4_4h
; Make sure we are using the operands defined by the ABI
; CHECK: ld4.4h { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x4x4_t @llvm.aarch64.neon.ld4.v4i16.p0i16(i16* %A)
	ret %struct.__neon_int16x4x4_t  %tmp2
}

declare %struct.__neon_int16x4x2_t @llvm.aarch64.neon.ld2.v4i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.aarch64.neon.ld3.v4i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int16x4x4_t @llvm.aarch64.neon.ld4.v4i16.p0i16(i16*) nounwind readonly

%struct.__neon_int16x8x2_t = type { <8 x i16>,  <8 x i16> }
%struct.__neon_int16x8x3_t = type { <8 x i16>,  <8 x i16>,  <8 x i16> }
%struct.__neon_int16x8x4_t = type { <8 x i16>,  <8 x i16>, <8 x i16>,  <8 x i16> }

define %struct.__neon_int16x8x2_t @ld2_8h(i16* %A) nounwind {
; CHECK-LABEL: ld2_8h
; Make sure we are using the operands defined by the ABI
; CHECK: ld2.8h { v0, v1 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int16x8x2_t @llvm.aarch64.neon.ld2.v8i16.p0i16(i16* %A)
  ret %struct.__neon_int16x8x2_t  %tmp2
}

define %struct.__neon_int16x8x3_t @ld3_8h(i16* %A) nounwind {
; CHECK-LABEL: ld3_8h
; Make sure we are using the operands defined by the ABI
; CHECK: ld3.8h { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int16x8x3_t @llvm.aarch64.neon.ld3.v8i16.p0i16(i16* %A)
  ret %struct.__neon_int16x8x3_t %tmp2
}

define %struct.__neon_int16x8x4_t @ld4_8h(i16* %A) nounwind {
; CHECK-LABEL: ld4_8h
; Make sure we are using the operands defined by the ABI
; CHECK: ld4.8h { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int16x8x4_t @llvm.aarch64.neon.ld4.v8i16.p0i16(i16* %A)
  ret %struct.__neon_int16x8x4_t  %tmp2
}

declare %struct.__neon_int16x8x2_t @llvm.aarch64.neon.ld2.v8i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int16x8x3_t @llvm.aarch64.neon.ld3.v8i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int16x8x4_t @llvm.aarch64.neon.ld4.v8i16.p0i16(i16*) nounwind readonly

%struct.__neon_int32x2x2_t = type { <2 x i32>,  <2 x i32> }
%struct.__neon_int32x2x3_t = type { <2 x i32>,  <2 x i32>,  <2 x i32> }
%struct.__neon_int32x2x4_t = type { <2 x i32>,  <2 x i32>, <2 x i32>,  <2 x i32> }

define %struct.__neon_int32x2x2_t @ld2_2s(i32* %A) nounwind {
; CHECK-LABEL: ld2_2s
; Make sure we are using the operands defined by the ABI
; CHECK: ld2.2s { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x2x2_t @llvm.aarch64.neon.ld2.v2i32.p0i32(i32* %A)
	ret %struct.__neon_int32x2x2_t  %tmp2
}

define %struct.__neon_int32x2x3_t @ld3_2s(i32* %A) nounwind {
; CHECK-LABEL: ld3_2s
; Make sure we are using the operands defined by the ABI
; CHECK: ld3.2s { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x2x3_t @llvm.aarch64.neon.ld3.v2i32.p0i32(i32* %A)
	ret %struct.__neon_int32x2x3_t  %tmp2
}

define %struct.__neon_int32x2x4_t @ld4_2s(i32* %A) nounwind {
; CHECK-LABEL: ld4_2s
; Make sure we are using the operands defined by the ABI
; CHECK: ld4.2s { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x2x4_t @llvm.aarch64.neon.ld4.v2i32.p0i32(i32* %A)
	ret %struct.__neon_int32x2x4_t  %tmp2
}

declare %struct.__neon_int32x2x2_t @llvm.aarch64.neon.ld2.v2i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.aarch64.neon.ld3.v2i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_int32x2x4_t @llvm.aarch64.neon.ld4.v2i32.p0i32(i32*) nounwind readonly

%struct.__neon_int32x4x2_t = type { <4 x i32>,  <4 x i32> }
%struct.__neon_int32x4x3_t = type { <4 x i32>,  <4 x i32>,  <4 x i32> }
%struct.__neon_int32x4x4_t = type { <4 x i32>,  <4 x i32>, <4 x i32>,  <4 x i32> }

define %struct.__neon_int32x4x2_t @ld2_4s(i32* %A) nounwind {
; CHECK-LABEL: ld2_4s
; Make sure we are using the operands defined by the ABI
; CHECK: ld2.4s { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x2_t @llvm.aarch64.neon.ld2.v4i32.p0i32(i32* %A)
	ret %struct.__neon_int32x4x2_t  %tmp2
}

define %struct.__neon_int32x4x3_t @ld3_4s(i32* %A) nounwind {
; CHECK-LABEL: ld3_4s
; Make sure we are using the operands defined by the ABI
; CHECK: ld3.4s { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x3_t @llvm.aarch64.neon.ld3.v4i32.p0i32(i32* %A)
	ret %struct.__neon_int32x4x3_t  %tmp2
}

define %struct.__neon_int32x4x4_t @ld4_4s(i32* %A) nounwind {
; CHECK-LABEL: ld4_4s
; Make sure we are using the operands defined by the ABI
; CHECK: ld4.4s { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x4_t @llvm.aarch64.neon.ld4.v4i32.p0i32(i32* %A)
	ret %struct.__neon_int32x4x4_t  %tmp2
}

declare %struct.__neon_int32x4x2_t @llvm.aarch64.neon.ld2.v4i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.aarch64.neon.ld3.v4i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_int32x4x4_t @llvm.aarch64.neon.ld4.v4i32.p0i32(i32*) nounwind readonly

%struct.__neon_int64x2x2_t = type { <2 x i64>,  <2 x i64> }
%struct.__neon_int64x2x3_t = type { <2 x i64>,  <2 x i64>,  <2 x i64> }
%struct.__neon_int64x2x4_t = type { <2 x i64>,  <2 x i64>, <2 x i64>,  <2 x i64> }

define %struct.__neon_int64x2x2_t @ld2_2d(i64* %A) nounwind {
; CHECK-LABEL: ld2_2d
; Make sure we are using the operands defined by the ABI
; CHECK: ld2.2d { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x2_t @llvm.aarch64.neon.ld2.v2i64.p0i64(i64* %A)
	ret %struct.__neon_int64x2x2_t  %tmp2
}

define %struct.__neon_int64x2x3_t @ld3_2d(i64* %A) nounwind {
; CHECK-LABEL: ld3_2d
; Make sure we are using the operands defined by the ABI
; CHECK: ld3.2d { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x3_t @llvm.aarch64.neon.ld3.v2i64.p0i64(i64* %A)
	ret %struct.__neon_int64x2x3_t  %tmp2
}

define %struct.__neon_int64x2x4_t @ld4_2d(i64* %A) nounwind {
; CHECK-LABEL: ld4_2d
; Make sure we are using the operands defined by the ABI
; CHECK: ld4.2d { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x4_t @llvm.aarch64.neon.ld4.v2i64.p0i64(i64* %A)
	ret %struct.__neon_int64x2x4_t  %tmp2
}

declare %struct.__neon_int64x2x2_t @llvm.aarch64.neon.ld2.v2i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_int64x2x3_t @llvm.aarch64.neon.ld3.v2i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_int64x2x4_t @llvm.aarch64.neon.ld4.v2i64.p0i64(i64*) nounwind readonly

%struct.__neon_int64x1x2_t = type { <1 x i64>,  <1 x i64> }
%struct.__neon_int64x1x3_t = type { <1 x i64>,  <1 x i64>, <1 x i64> }
%struct.__neon_int64x1x4_t = type { <1 x i64>,  <1 x i64>, <1 x i64>, <1 x i64> }


define %struct.__neon_int64x1x2_t @ld2_1di64(i64* %A) nounwind {
; CHECK-LABEL: ld2_1di64
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.1d { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x1x2_t @llvm.aarch64.neon.ld2.v1i64.p0i64(i64* %A)
	ret %struct.__neon_int64x1x2_t  %tmp2
}

define %struct.__neon_int64x1x3_t @ld3_1di64(i64* %A) nounwind {
; CHECK-LABEL: ld3_1di64
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.1d { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x1x3_t @llvm.aarch64.neon.ld3.v1i64.p0i64(i64* %A)
	ret %struct.__neon_int64x1x3_t  %tmp2
}

define %struct.__neon_int64x1x4_t @ld4_1di64(i64* %A) nounwind {
; CHECK-LABEL: ld4_1di64
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.1d { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x1x4_t @llvm.aarch64.neon.ld4.v1i64.p0i64(i64* %A)
	ret %struct.__neon_int64x1x4_t  %tmp2
}


declare %struct.__neon_int64x1x2_t @llvm.aarch64.neon.ld2.v1i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_int64x1x3_t @llvm.aarch64.neon.ld3.v1i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_int64x1x4_t @llvm.aarch64.neon.ld4.v1i64.p0i64(i64*) nounwind readonly

%struct.__neon_float64x1x2_t = type { <1 x double>,  <1 x double> }
%struct.__neon_float64x1x3_t = type { <1 x double>,  <1 x double>, <1 x double> }
%struct.__neon_float64x1x4_t = type { <1 x double>,  <1 x double>, <1 x double>, <1 x double> }


define %struct.__neon_float64x1x2_t @ld2_1df64(double* %A) nounwind {
; CHECK-LABEL: ld2_1df64
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.1d { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_float64x1x2_t @llvm.aarch64.neon.ld2.v1f64.p0f64(double* %A)
	ret %struct.__neon_float64x1x2_t  %tmp2
}

define %struct.__neon_float64x1x3_t @ld3_1df64(double* %A) nounwind {
; CHECK-LABEL: ld3_1df64
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.1d { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_float64x1x3_t @llvm.aarch64.neon.ld3.v1f64.p0f64(double* %A)
	ret %struct.__neon_float64x1x3_t  %tmp2
}

define %struct.__neon_float64x1x4_t @ld4_1df64(double* %A) nounwind {
; CHECK-LABEL: ld4_1df64
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.1d { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_float64x1x4_t @llvm.aarch64.neon.ld4.v1f64.p0f64(double* %A)
	ret %struct.__neon_float64x1x4_t  %tmp2
}

declare %struct.__neon_float64x1x2_t @llvm.aarch64.neon.ld2.v1f64.p0f64(double*) nounwind readonly
declare %struct.__neon_float64x1x3_t @llvm.aarch64.neon.ld3.v1f64.p0f64(double*) nounwind readonly
declare %struct.__neon_float64x1x4_t @llvm.aarch64.neon.ld4.v1f64.p0f64(double*) nounwind readonly


define %struct.__neon_int8x16x2_t @ld2lane_16b(<16 x i8> %L1, <16 x i8> %L2, i8* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld2lane_16b
; CHECK: ld2.b { v0, v1 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x16x2_t @llvm.aarch64.neon.ld2lane.v16i8.p0i8(<16 x i8> %L1, <16 x i8> %L2, i64 1, i8* %A)
	ret %struct.__neon_int8x16x2_t  %tmp2
}

define %struct.__neon_int8x16x3_t @ld3lane_16b(<16 x i8> %L1, <16 x i8> %L2, <16 x i8> %L3, i8* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld3lane_16b
; CHECK: ld3.b { v0, v1, v2 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x16x3_t @llvm.aarch64.neon.ld3lane.v16i8.p0i8(<16 x i8> %L1, <16 x i8> %L2, <16 x i8> %L3, i64 1, i8* %A)
	ret %struct.__neon_int8x16x3_t  %tmp2
}

define %struct.__neon_int8x16x4_t @ld4lane_16b(<16 x i8> %L1, <16 x i8> %L2, <16 x i8> %L3, <16 x i8> %L4, i8* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld4lane_16b
; CHECK: ld4.b { v0, v1, v2, v3 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x16x4_t @llvm.aarch64.neon.ld4lane.v16i8.p0i8(<16 x i8> %L1, <16 x i8> %L2, <16 x i8> %L3, <16 x i8> %L4, i64 1, i8* %A)
	ret %struct.__neon_int8x16x4_t  %tmp2
}

declare %struct.__neon_int8x16x2_t @llvm.aarch64.neon.ld2lane.v16i8.p0i8(<16 x i8>, <16 x i8>, i64, i8*) nounwind readonly
declare %struct.__neon_int8x16x3_t @llvm.aarch64.neon.ld3lane.v16i8.p0i8(<16 x i8>, <16 x i8>, <16 x i8>, i64, i8*) nounwind readonly
declare %struct.__neon_int8x16x4_t @llvm.aarch64.neon.ld4lane.v16i8.p0i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i64, i8*) nounwind readonly

define %struct.__neon_int16x8x2_t @ld2lane_8h(<8 x i16> %L1, <8 x i16> %L2, i16* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld2lane_8h
; CHECK: ld2.h { v0, v1 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x8x2_t @llvm.aarch64.neon.ld2lane.v8i16.p0i16(<8 x i16> %L1, <8 x i16> %L2, i64 1, i16* %A)
	ret %struct.__neon_int16x8x2_t  %tmp2
}

define %struct.__neon_int16x8x3_t @ld3lane_8h(<8 x i16> %L1, <8 x i16> %L2, <8 x i16> %L3, i16* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld3lane_8h
; CHECK: ld3.h { v0, v1, v2 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x8x3_t @llvm.aarch64.neon.ld3lane.v8i16.p0i16(<8 x i16> %L1, <8 x i16> %L2, <8 x i16> %L3, i64 1, i16* %A)
	ret %struct.__neon_int16x8x3_t  %tmp2
}

define %struct.__neon_int16x8x4_t @ld4lane_8h(<8 x i16> %L1, <8 x i16> %L2, <8 x i16> %L3, <8 x i16> %L4, i16* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld4lane_8h
; CHECK: ld4.h { v0, v1, v2, v3 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x8x4_t @llvm.aarch64.neon.ld4lane.v8i16.p0i16(<8 x i16> %L1, <8 x i16> %L2, <8 x i16> %L3, <8 x i16> %L4, i64 1, i16* %A)
	ret %struct.__neon_int16x8x4_t  %tmp2
}

declare %struct.__neon_int16x8x2_t @llvm.aarch64.neon.ld2lane.v8i16.p0i16(<8 x i16>, <8 x i16>, i64, i16*) nounwind readonly
declare %struct.__neon_int16x8x3_t @llvm.aarch64.neon.ld3lane.v8i16.p0i16(<8 x i16>, <8 x i16>, <8 x i16>, i64, i16*) nounwind readonly
declare %struct.__neon_int16x8x4_t @llvm.aarch64.neon.ld4lane.v8i16.p0i16(<8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i64, i16*) nounwind readonly

define %struct.__neon_int32x4x2_t @ld2lane_4s(<4 x i32> %L1, <4 x i32> %L2, i32* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld2lane_4s
; CHECK: ld2.s { v0, v1 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x2_t @llvm.aarch64.neon.ld2lane.v4i32.p0i32(<4 x i32> %L1, <4 x i32> %L2, i64 1, i32* %A)
	ret %struct.__neon_int32x4x2_t  %tmp2
}

define %struct.__neon_int32x4x3_t @ld3lane_4s(<4 x i32> %L1, <4 x i32> %L2, <4 x i32> %L3, i32* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld3lane_4s
; CHECK: ld3.s { v0, v1, v2 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x3_t @llvm.aarch64.neon.ld3lane.v4i32.p0i32(<4 x i32> %L1, <4 x i32> %L2, <4 x i32> %L3, i64 1, i32* %A)
	ret %struct.__neon_int32x4x3_t  %tmp2
}

define %struct.__neon_int32x4x4_t @ld4lane_4s(<4 x i32> %L1, <4 x i32> %L2, <4 x i32> %L3, <4 x i32> %L4, i32* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld4lane_4s
; CHECK: ld4.s { v0, v1, v2, v3 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x4_t @llvm.aarch64.neon.ld4lane.v4i32.p0i32(<4 x i32> %L1, <4 x i32> %L2, <4 x i32> %L3, <4 x i32> %L4, i64 1, i32* %A)
	ret %struct.__neon_int32x4x4_t  %tmp2
}

declare %struct.__neon_int32x4x2_t @llvm.aarch64.neon.ld2lane.v4i32.p0i32(<4 x i32>, <4 x i32>, i64, i32*) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.aarch64.neon.ld3lane.v4i32.p0i32(<4 x i32>, <4 x i32>, <4 x i32>, i64, i32*) nounwind readonly
declare %struct.__neon_int32x4x4_t @llvm.aarch64.neon.ld4lane.v4i32.p0i32(<4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i64, i32*) nounwind readonly

define %struct.__neon_int64x2x2_t @ld2lane_2d(<2 x i64> %L1, <2 x i64> %L2, i64* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld2lane_2d
; CHECK: ld2.d { v0, v1 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x2_t @llvm.aarch64.neon.ld2lane.v2i64.p0i64(<2 x i64> %L1, <2 x i64> %L2, i64 1, i64* %A)
	ret %struct.__neon_int64x2x2_t  %tmp2
}

define %struct.__neon_int64x2x3_t @ld3lane_2d(<2 x i64> %L1, <2 x i64> %L2, <2 x i64> %L3, i64* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld3lane_2d
; CHECK: ld3.d { v0, v1, v2 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x3_t @llvm.aarch64.neon.ld3lane.v2i64.p0i64(<2 x i64> %L1, <2 x i64> %L2, <2 x i64> %L3, i64 1, i64* %A)
	ret %struct.__neon_int64x2x3_t  %tmp2
}

define %struct.__neon_int64x2x4_t @ld4lane_2d(<2 x i64> %L1, <2 x i64> %L2, <2 x i64> %L3, <2 x i64> %L4, i64* %A) nounwind {
; Make sure we are using the operands defined by the ABI
; CHECK: ld4lane_2d
; CHECK: ld4.d { v0, v1, v2, v3 }[1], [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x4_t @llvm.aarch64.neon.ld4lane.v2i64.p0i64(<2 x i64> %L1, <2 x i64> %L2, <2 x i64> %L3, <2 x i64> %L4, i64 1, i64* %A)
	ret %struct.__neon_int64x2x4_t  %tmp2
}

declare %struct.__neon_int64x2x2_t @llvm.aarch64.neon.ld2lane.v2i64.p0i64(<2 x i64>, <2 x i64>, i64, i64*) nounwind readonly
declare %struct.__neon_int64x2x3_t @llvm.aarch64.neon.ld3lane.v2i64.p0i64(<2 x i64>, <2 x i64>, <2 x i64>, i64, i64*) nounwind readonly
declare %struct.__neon_int64x2x4_t @llvm.aarch64.neon.ld4lane.v2i64.p0i64(<2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, i64, i64*) nounwind readonly

define <8 x i8> @ld1r_8b(i8* %bar) {
; CHECK: ld1r_8b
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.8b { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp1 = load i8, i8* %bar
  %tmp2 = insertelement <8 x i8> <i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef>, i8 %tmp1, i32 0
  %tmp3 = insertelement <8 x i8> %tmp2, i8 %tmp1, i32 1
  %tmp4 = insertelement <8 x i8> %tmp3, i8 %tmp1, i32 2
  %tmp5 = insertelement <8 x i8> %tmp4, i8 %tmp1, i32 3
  %tmp6 = insertelement <8 x i8> %tmp5, i8 %tmp1, i32 4
  %tmp7 = insertelement <8 x i8> %tmp6, i8 %tmp1, i32 5
  %tmp8 = insertelement <8 x i8> %tmp7, i8 %tmp1, i32 6
  %tmp9 = insertelement <8 x i8> %tmp8, i8 %tmp1, i32 7
  ret <8 x i8> %tmp9
}

define <16 x i8> @ld1r_16b(i8* %bar) {
; CHECK: ld1r_16b
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.16b { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp1 = load i8, i8* %bar
  %tmp2 = insertelement <16 x i8> <i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef>, i8 %tmp1, i32 0
  %tmp3 = insertelement <16 x i8> %tmp2, i8 %tmp1, i32 1
  %tmp4 = insertelement <16 x i8> %tmp3, i8 %tmp1, i32 2
  %tmp5 = insertelement <16 x i8> %tmp4, i8 %tmp1, i32 3
  %tmp6 = insertelement <16 x i8> %tmp5, i8 %tmp1, i32 4
  %tmp7 = insertelement <16 x i8> %tmp6, i8 %tmp1, i32 5
  %tmp8 = insertelement <16 x i8> %tmp7, i8 %tmp1, i32 6
  %tmp9 = insertelement <16 x i8> %tmp8, i8 %tmp1, i32 7
  %tmp10 = insertelement <16 x i8> %tmp9, i8 %tmp1, i32 8
  %tmp11 = insertelement <16 x i8> %tmp10, i8 %tmp1, i32 9
  %tmp12 = insertelement <16 x i8> %tmp11, i8 %tmp1, i32 10
  %tmp13 = insertelement <16 x i8> %tmp12, i8 %tmp1, i32 11
  %tmp14 = insertelement <16 x i8> %tmp13, i8 %tmp1, i32 12
  %tmp15 = insertelement <16 x i8> %tmp14, i8 %tmp1, i32 13
  %tmp16 = insertelement <16 x i8> %tmp15, i8 %tmp1, i32 14
  %tmp17 = insertelement <16 x i8> %tmp16, i8 %tmp1, i32 15
  ret <16 x i8> %tmp17
}

define <4 x i16> @ld1r_4h(i16* %bar) {
; CHECK: ld1r_4h
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.4h { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp1 = load i16, i16* %bar
  %tmp2 = insertelement <4 x i16> <i16 undef, i16 undef, i16 undef, i16 undef>, i16 %tmp1, i32 0
  %tmp3 = insertelement <4 x i16> %tmp2, i16 %tmp1, i32 1
  %tmp4 = insertelement <4 x i16> %tmp3, i16 %tmp1, i32 2
  %tmp5 = insertelement <4 x i16> %tmp4, i16 %tmp1, i32 3
  ret <4 x i16> %tmp5
}

define <8 x i16> @ld1r_8h(i16* %bar) {
; CHECK: ld1r_8h
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.8h { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp1 = load i16, i16* %bar
  %tmp2 = insertelement <8 x i16> <i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef>, i16 %tmp1, i32 0
  %tmp3 = insertelement <8 x i16> %tmp2, i16 %tmp1, i32 1
  %tmp4 = insertelement <8 x i16> %tmp3, i16 %tmp1, i32 2
  %tmp5 = insertelement <8 x i16> %tmp4, i16 %tmp1, i32 3
  %tmp6 = insertelement <8 x i16> %tmp5, i16 %tmp1, i32 4
  %tmp7 = insertelement <8 x i16> %tmp6, i16 %tmp1, i32 5
  %tmp8 = insertelement <8 x i16> %tmp7, i16 %tmp1, i32 6
  %tmp9 = insertelement <8 x i16> %tmp8, i16 %tmp1, i32 7
  ret <8 x i16> %tmp9
}

define <2 x i32> @ld1r_2s(i32* %bar) {
; CHECK: ld1r_2s
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.2s { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp1 = load i32, i32* %bar
  %tmp2 = insertelement <2 x i32> <i32 undef, i32 undef>, i32 %tmp1, i32 0
  %tmp3 = insertelement <2 x i32> %tmp2, i32 %tmp1, i32 1
  ret <2 x i32> %tmp3
}

define <4 x i32> @ld1r_4s(i32* %bar) {
; CHECK: ld1r_4s
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.4s { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp1 = load i32, i32* %bar
  %tmp2 = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %tmp1, i32 0
  %tmp3 = insertelement <4 x i32> %tmp2, i32 %tmp1, i32 1
  %tmp4 = insertelement <4 x i32> %tmp3, i32 %tmp1, i32 2
  %tmp5 = insertelement <4 x i32> %tmp4, i32 %tmp1, i32 3
  ret <4 x i32> %tmp5
}

define <2 x i64> @ld1r_2d(i64* %bar) {
; CHECK: ld1r_2d
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.2d { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp1 = load i64, i64* %bar
  %tmp2 = insertelement <2 x i64> <i64 undef, i64 undef>, i64 %tmp1, i32 0
  %tmp3 = insertelement <2 x i64> %tmp2, i64 %tmp1, i32 1
  ret <2 x i64> %tmp3
}

define %struct.__neon_int8x8x2_t @ld2r_8b(i8* %A) nounwind {
; CHECK: ld2r_8b
; Make sure we are using the operands defined by the ABI
; CHECK: ld2r.8b { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x8x2_t @llvm.aarch64.neon.ld2r.v8i8.p0i8(i8* %A)
	ret %struct.__neon_int8x8x2_t  %tmp2
}

define %struct.__neon_int8x8x3_t @ld3r_8b(i8* %A) nounwind {
; CHECK: ld3r_8b
; Make sure we are using the operands defined by the ABI
; CHECK: ld3r.8b { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x8x3_t @llvm.aarch64.neon.ld3r.v8i8.p0i8(i8* %A)
	ret %struct.__neon_int8x8x3_t  %tmp2
}

define %struct.__neon_int8x8x4_t @ld4r_8b(i8* %A) nounwind {
; CHECK: ld4r_8b
; Make sure we are using the operands defined by the ABI
; CHECK: ld4r.8b { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x8x4_t @llvm.aarch64.neon.ld4r.v8i8.p0i8(i8* %A)
	ret %struct.__neon_int8x8x4_t  %tmp2
}

declare %struct.__neon_int8x8x2_t @llvm.aarch64.neon.ld2r.v8i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int8x8x3_t @llvm.aarch64.neon.ld3r.v8i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int8x8x4_t @llvm.aarch64.neon.ld4r.v8i8.p0i8(i8*) nounwind readonly

define %struct.__neon_int8x16x2_t @ld2r_16b(i8* %A) nounwind {
; CHECK: ld2r_16b
; Make sure we are using the operands defined by the ABI
; CHECK: ld2r.16b { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x16x2_t @llvm.aarch64.neon.ld2r.v16i8.p0i8(i8* %A)
	ret %struct.__neon_int8x16x2_t  %tmp2
}

define %struct.__neon_int8x16x3_t @ld3r_16b(i8* %A) nounwind {
; CHECK: ld3r_16b
; Make sure we are using the operands defined by the ABI
; CHECK: ld3r.16b { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x16x3_t @llvm.aarch64.neon.ld3r.v16i8.p0i8(i8* %A)
	ret %struct.__neon_int8x16x3_t  %tmp2
}

define %struct.__neon_int8x16x4_t @ld4r_16b(i8* %A) nounwind {
; CHECK: ld4r_16b
; Make sure we are using the operands defined by the ABI
; CHECK: ld4r.16b { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int8x16x4_t @llvm.aarch64.neon.ld4r.v16i8.p0i8(i8* %A)
	ret %struct.__neon_int8x16x4_t  %tmp2
}

declare %struct.__neon_int8x16x2_t @llvm.aarch64.neon.ld2r.v16i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int8x16x3_t @llvm.aarch64.neon.ld3r.v16i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int8x16x4_t @llvm.aarch64.neon.ld4r.v16i8.p0i8(i8*) nounwind readonly

define %struct.__neon_int16x4x2_t @ld2r_4h(i16* %A) nounwind {
; CHECK: ld2r_4h
; Make sure we are using the operands defined by the ABI
; CHECK: ld2r.4h { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x4x2_t @llvm.aarch64.neon.ld2r.v4i16.p0i16(i16* %A)
	ret %struct.__neon_int16x4x2_t  %tmp2
}

define %struct.__neon_int16x4x3_t @ld3r_4h(i16* %A) nounwind {
; CHECK: ld3r_4h
; Make sure we are using the operands defined by the ABI
; CHECK: ld3r.4h { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x4x3_t @llvm.aarch64.neon.ld3r.v4i16.p0i16(i16* %A)
	ret %struct.__neon_int16x4x3_t  %tmp2
}

define %struct.__neon_int16x4x4_t @ld4r_4h(i16* %A) nounwind {
; CHECK: ld4r_4h
; Make sure we are using the operands defined by the ABI
; CHECK: ld4r.4h { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int16x4x4_t @llvm.aarch64.neon.ld4r.v4i16.p0i16(i16* %A)
	ret %struct.__neon_int16x4x4_t  %tmp2
}

declare %struct.__neon_int16x4x2_t @llvm.aarch64.neon.ld2r.v4i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.aarch64.neon.ld3r.v4i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int16x4x4_t @llvm.aarch64.neon.ld4r.v4i16.p0i16(i16*) nounwind readonly

define %struct.__neon_int16x8x2_t @ld2r_8h(i16* %A) nounwind {
; CHECK: ld2r_8h
; Make sure we are using the operands defined by the ABI
; CHECK: ld2r.8h { v0, v1 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int16x8x2_t @llvm.aarch64.neon.ld2r.v8i16.p0i16(i16* %A)
  ret %struct.__neon_int16x8x2_t  %tmp2
}

define %struct.__neon_int16x8x3_t @ld3r_8h(i16* %A) nounwind {
; CHECK: ld3r_8h
; Make sure we are using the operands defined by the ABI
; CHECK: ld3r.8h { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int16x8x3_t @llvm.aarch64.neon.ld3r.v8i16.p0i16(i16* %A)
  ret %struct.__neon_int16x8x3_t  %tmp2
}

define %struct.__neon_int16x8x4_t @ld4r_8h(i16* %A) nounwind {
; CHECK: ld4r_8h
; Make sure we are using the operands defined by the ABI
; CHECK: ld4r.8h { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
  %tmp2 = call %struct.__neon_int16x8x4_t @llvm.aarch64.neon.ld4r.v8i16.p0i16(i16* %A)
  ret %struct.__neon_int16x8x4_t  %tmp2
}

declare %struct.__neon_int16x8x2_t @llvm.aarch64.neon.ld2r.v8i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int16x8x3_t @llvm.aarch64.neon.ld3r.v8i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int16x8x4_t @llvm.aarch64.neon.ld4r.v8i16.p0i16(i16*) nounwind readonly

define %struct.__neon_int32x2x2_t @ld2r_2s(i32* %A) nounwind {
; CHECK: ld2r_2s
; Make sure we are using the operands defined by the ABI
; CHECK: ld2r.2s { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x2x2_t @llvm.aarch64.neon.ld2r.v2i32.p0i32(i32* %A)
	ret %struct.__neon_int32x2x2_t  %tmp2
}

define %struct.__neon_int32x2x3_t @ld3r_2s(i32* %A) nounwind {
; CHECK: ld3r_2s
; Make sure we are using the operands defined by the ABI
; CHECK: ld3r.2s { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x2x3_t @llvm.aarch64.neon.ld3r.v2i32.p0i32(i32* %A)
	ret %struct.__neon_int32x2x3_t  %tmp2
}

define %struct.__neon_int32x2x4_t @ld4r_2s(i32* %A) nounwind {
; CHECK: ld4r_2s
; Make sure we are using the operands defined by the ABI
; CHECK: ld4r.2s { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x2x4_t @llvm.aarch64.neon.ld4r.v2i32.p0i32(i32* %A)
	ret %struct.__neon_int32x2x4_t  %tmp2
}

declare %struct.__neon_int32x2x2_t @llvm.aarch64.neon.ld2r.v2i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.aarch64.neon.ld3r.v2i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_int32x2x4_t @llvm.aarch64.neon.ld4r.v2i32.p0i32(i32*) nounwind readonly

define %struct.__neon_int32x4x2_t @ld2r_4s(i32* %A) nounwind {
; CHECK: ld2r_4s
; Make sure we are using the operands defined by the ABI
; CHECK: ld2r.4s { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x2_t @llvm.aarch64.neon.ld2r.v4i32.p0i32(i32* %A)
	ret %struct.__neon_int32x4x2_t  %tmp2
}

define %struct.__neon_int32x4x3_t @ld3r_4s(i32* %A) nounwind {
; CHECK: ld3r_4s
; Make sure we are using the operands defined by the ABI
; CHECK: ld3r.4s { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x3_t @llvm.aarch64.neon.ld3r.v4i32.p0i32(i32* %A)
	ret %struct.__neon_int32x4x3_t  %tmp2
}

define %struct.__neon_int32x4x4_t @ld4r_4s(i32* %A) nounwind {
; CHECK: ld4r_4s
; Make sure we are using the operands defined by the ABI
; CHECK: ld4r.4s { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int32x4x4_t @llvm.aarch64.neon.ld4r.v4i32.p0i32(i32* %A)
	ret %struct.__neon_int32x4x4_t  %tmp2
}

declare %struct.__neon_int32x4x2_t @llvm.aarch64.neon.ld2r.v4i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.aarch64.neon.ld3r.v4i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_int32x4x4_t @llvm.aarch64.neon.ld4r.v4i32.p0i32(i32*) nounwind readonly

define %struct.__neon_int64x1x2_t @ld2r_1d(i64* %A) nounwind {
; CHECK: ld2r_1d
; Make sure we are using the operands defined by the ABI
; CHECK: ld2r.1d { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x1x2_t @llvm.aarch64.neon.ld2r.v1i64.p0i64(i64* %A)
	ret %struct.__neon_int64x1x2_t  %tmp2
}

define %struct.__neon_int64x1x3_t @ld3r_1d(i64* %A) nounwind {
; CHECK: ld3r_1d
; Make sure we are using the operands defined by the ABI
; CHECK: ld3r.1d { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x1x3_t @llvm.aarch64.neon.ld3r.v1i64.p0i64(i64* %A)
	ret %struct.__neon_int64x1x3_t  %tmp2
}

define %struct.__neon_int64x1x4_t @ld4r_1d(i64* %A) nounwind {
; CHECK: ld4r_1d
; Make sure we are using the operands defined by the ABI
; CHECK: ld4r.1d { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x1x4_t @llvm.aarch64.neon.ld4r.v1i64.p0i64(i64* %A)
	ret %struct.__neon_int64x1x4_t  %tmp2
}

declare %struct.__neon_int64x1x2_t @llvm.aarch64.neon.ld2r.v1i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_int64x1x3_t @llvm.aarch64.neon.ld3r.v1i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_int64x1x4_t @llvm.aarch64.neon.ld4r.v1i64.p0i64(i64*) nounwind readonly

define %struct.__neon_int64x2x2_t @ld2r_2d(i64* %A) nounwind {
; CHECK: ld2r_2d
; Make sure we are using the operands defined by the ABI
; CHECK: ld2r.2d { v0, v1 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x2_t @llvm.aarch64.neon.ld2r.v2i64.p0i64(i64* %A)
	ret %struct.__neon_int64x2x2_t  %tmp2
}

define %struct.__neon_int64x2x3_t @ld3r_2d(i64* %A) nounwind {
; CHECK: ld3r_2d
; Make sure we are using the operands defined by the ABI
; CHECK: ld3r.2d { v0, v1, v2 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x3_t @llvm.aarch64.neon.ld3r.v2i64.p0i64(i64* %A)
	ret %struct.__neon_int64x2x3_t  %tmp2
}

define %struct.__neon_int64x2x4_t @ld4r_2d(i64* %A) nounwind {
; CHECK: ld4r_2d
; Make sure we are using the operands defined by the ABI
; CHECK: ld4r.2d { v0, v1, v2, v3 }, [x0]
; CHECK-NEXT: ret
	%tmp2 = call %struct.__neon_int64x2x4_t @llvm.aarch64.neon.ld4r.v2i64.p0i64(i64* %A)
	ret %struct.__neon_int64x2x4_t  %tmp2
}

declare %struct.__neon_int64x2x2_t @llvm.aarch64.neon.ld2r.v2i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_int64x2x3_t @llvm.aarch64.neon.ld3r.v2i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_int64x2x4_t @llvm.aarch64.neon.ld4r.v2i64.p0i64(i64*) nounwind readonly

define <16 x i8> @ld1_16b(<16 x i8> %V, i8* %bar) {
; CHECK-LABEL: ld1_16b
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.b { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load i8, i8* %bar
  %tmp2 = insertelement <16 x i8> %V, i8 %tmp1, i32 0
  ret <16 x i8> %tmp2
}

define <8 x i16> @ld1_8h(<8 x i16> %V, i16* %bar) {
; CHECK-LABEL: ld1_8h
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.h { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load i16, i16* %bar
  %tmp2 = insertelement <8 x i16> %V, i16 %tmp1, i32 0
  ret <8 x i16> %tmp2
}

define <4 x i32> @ld1_4s(<4 x i32> %V, i32* %bar) {
; CHECK-LABEL: ld1_4s
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.s { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load i32, i32* %bar
  %tmp2 = insertelement <4 x i32> %V, i32 %tmp1, i32 0
  ret <4 x i32> %tmp2
}

define <4 x float> @ld1_4s_float(<4 x float> %V, float* %bar) {
; CHECK-LABEL: ld1_4s_float:
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.s { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load float, float* %bar
  %tmp2 = insertelement <4 x float> %V, float %tmp1, i32 0
  ret <4 x float> %tmp2
}

define <2 x i64> @ld1_2d(<2 x i64> %V, i64* %bar) {
; CHECK-LABEL: ld1_2d
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.d { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load i64, i64* %bar
  %tmp2 = insertelement <2 x i64> %V, i64 %tmp1, i32 0
  ret <2 x i64> %tmp2
}

define <2 x double> @ld1_2d_double(<2 x double> %V, double* %bar) {
; CHECK-LABEL: ld1_2d_double:
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.d { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load double, double* %bar
  %tmp2 = insertelement <2 x double> %V, double %tmp1, i32 0
  ret <2 x double> %tmp2
}

define <1 x i64> @ld1_1d(<1 x i64>* %p) {
; CHECK-LABEL: ld1_1d
; Make sure we are using the operands defined by the ABI
; CHECK: ldr [[REG:d[0-9]+]], [x0]
; CHECK-NEXT: ret
  %tmp = load <1 x i64>, <1 x i64>* %p, align 8
  ret <1 x i64> %tmp
}

define <8 x i8> @ld1_8b(<8 x i8> %V, i8* %bar) {
; CHECK-LABEL: ld1_8b
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.b { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load i8, i8* %bar
  %tmp2 = insertelement <8 x i8> %V, i8 %tmp1, i32 0
  ret <8 x i8> %tmp2
}

define <4 x i16> @ld1_4h(<4 x i16> %V, i16* %bar) {
; CHECK-LABEL: ld1_4h
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.h { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load i16, i16* %bar
  %tmp2 = insertelement <4 x i16> %V, i16 %tmp1, i32 0
  ret <4 x i16> %tmp2
}

define <2 x i32> @ld1_2s(<2 x i32> %V, i32* %bar) {
; CHECK-LABEL: ld1_2s:
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.s { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load i32, i32* %bar
  %tmp2 = insertelement <2 x i32> %V, i32 %tmp1, i32 0
  ret <2 x i32> %tmp2
}

define <2 x float> @ld1_2s_float(<2 x float> %V, float* %bar) {
; CHECK-LABEL: ld1_2s_float:
; Make sure we are using the operands defined by the ABI
; CHECK: ld1.s { v0 }[0], [x0]
; CHECK-NEXT: ret
  %tmp1 = load float, float* %bar
  %tmp2 = insertelement <2 x float> %V, float %tmp1, i32 0
  ret <2 x float> %tmp2
}


; Add rdar://13098923 test case: vld1_dup_u32 doesn't generate ld1r.2s
define void @ld1r_2s_from_dup(i8* nocapture %a, i8* nocapture %b, i16* nocapture %diff) nounwind ssp {
entry:
; CHECK: ld1r_2s_from_dup
; CHECK: ld1r.2s { [[ARG1:v[0-9]+]] }, [x0]
; CHECK-NEXT: ld1r.2s { [[ARG2:v[0-9]+]] }, [x1]
; CHECK-NEXT: ushll.8h [[ARG1]], [[ARG1]], #0
; CHECK-NEXT: ushll.8h [[ARG2]], [[ARG2]], #0
; CHECK-NEXT: sub.4h v[[RESREGNUM:[0-9]+]], [[ARG1]], [[ARG2]]
; CHECK-NEXT: str d[[RESREGNUM]], [x2]
; CHECK-NEXT: ret
  %tmp = bitcast i8* %a to i32*
  %tmp1 = load i32, i32* %tmp, align 4
  %tmp2 = insertelement <2 x i32> undef, i32 %tmp1, i32 0
  %lane = shufflevector <2 x i32> %tmp2, <2 x i32> undef, <2 x i32> zeroinitializer
  %tmp3 = bitcast <2 x i32> %lane to <8 x i8>
  %tmp4 = bitcast i8* %b to i32*
  %tmp5 = load i32, i32* %tmp4, align 4
  %tmp6 = insertelement <2 x i32> undef, i32 %tmp5, i32 0
  %lane1 = shufflevector <2 x i32> %tmp6, <2 x i32> undef, <2 x i32> zeroinitializer
  %tmp7 = bitcast <2 x i32> %lane1 to <8 x i8>
  %vmovl.i.i = zext <8 x i8> %tmp3 to <8 x i16>
  %vmovl.i4.i = zext <8 x i8> %tmp7 to <8 x i16>
  %sub.i = sub <8 x i16> %vmovl.i.i, %vmovl.i4.i
  %tmp8 = bitcast <8 x i16> %sub.i to <2 x i64>
  %shuffle.i = shufflevector <2 x i64> %tmp8, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp9 = bitcast <1 x i64> %shuffle.i to <4 x i16>
  %tmp10 = bitcast i16* %diff to <4 x i16>*
  store <4 x i16> %tmp9, <4 x i16>* %tmp10, align 8
  ret void
}

; Tests for rdar://11947069: vld1_dup_* and vld1q_dup_* code gen is suboptimal
define <4 x float> @ld1r_4s_float(float* nocapture %x) {
entry:
; CHECK-LABEL: ld1r_4s_float
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.4s { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp = load float, float* %x, align 4
  %tmp1 = insertelement <4 x float> undef, float %tmp, i32 0
  %tmp2 = insertelement <4 x float> %tmp1, float %tmp, i32 1
  %tmp3 = insertelement <4 x float> %tmp2, float %tmp, i32 2
  %tmp4 = insertelement <4 x float> %tmp3, float %tmp, i32 3
  ret <4 x float> %tmp4
}

define <2 x float> @ld1r_2s_float(float* nocapture %x) {
entry:
; CHECK-LABEL: ld1r_2s_float
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.2s { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp = load float, float* %x, align 4
  %tmp1 = insertelement <2 x float> undef, float %tmp, i32 0
  %tmp2 = insertelement <2 x float> %tmp1, float %tmp, i32 1
  ret <2 x float> %tmp2
}

define <2 x double> @ld1r_2d_double(double* nocapture %x) {
entry:
; CHECK-LABEL: ld1r_2d_double
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.2d { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp = load double, double* %x, align 4
  %tmp1 = insertelement <2 x double> undef, double %tmp, i32 0
  %tmp2 = insertelement <2 x double> %tmp1, double %tmp, i32 1
  ret <2 x double> %tmp2
}

define <1 x double> @ld1r_1d_double(double* nocapture %x) {
entry:
; CHECK-LABEL: ld1r_1d_double
; Make sure we are using the operands defined by the ABI
; CHECK: ldr d0, [x0]
; CHECK-NEXT: ret
  %tmp = load double, double* %x, align 4
  %tmp1 = insertelement <1 x double> undef, double %tmp, i32 0
  ret <1 x double> %tmp1
}

define <4 x float> @ld1r_4s_float_shuff(float* nocapture %x) {
entry:
; CHECK-LABEL: ld1r_4s_float_shuff
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.4s { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp = load float, float* %x, align 4
  %tmp1 = insertelement <4 x float> undef, float %tmp, i32 0
  %lane = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %lane
}

define <2 x float> @ld1r_2s_float_shuff(float* nocapture %x) {
entry:
; CHECK-LABEL: ld1r_2s_float_shuff
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.2s { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp = load float, float* %x, align 4
  %tmp1 = insertelement <2 x float> undef, float %tmp, i32 0
  %lane = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> zeroinitializer
  ret <2 x float> %lane
}

define <2 x double> @ld1r_2d_double_shuff(double* nocapture %x) {
entry:
; CHECK-LABEL: ld1r_2d_double_shuff
; Make sure we are using the operands defined by the ABI
; CHECK: ld1r.2d { v0 }, [x0]
; CHECK-NEXT: ret
  %tmp = load double, double* %x, align 4
  %tmp1 = insertelement <2 x double> undef, double %tmp, i32 0
  %lane = shufflevector <2 x double> %tmp1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %lane
}

define <1 x double> @ld1r_1d_double_shuff(double* nocapture %x) {
entry:
; CHECK-LABEL: ld1r_1d_double_shuff
; Make sure we are using the operands defined by the ABI
; CHECK: ldr d0, [x0]
; CHECK-NEXT: ret
  %tmp = load double, double* %x, align 4
  %tmp1 = insertelement <1 x double> undef, double %tmp, i32 0
  %lane = shufflevector <1 x double> %tmp1, <1 x double> undef, <1 x i32> zeroinitializer
  ret <1 x double> %lane
}

%struct.__neon_float32x2x2_t = type { <2 x float>,  <2 x float> }
%struct.__neon_float32x2x3_t = type { <2 x float>,  <2 x float>,  <2 x float> }
%struct.__neon_float32x2x4_t = type { <2 x float>,  <2 x float>, <2 x float>,  <2 x float> }

declare %struct.__neon_int8x8x2_t @llvm.aarch64.neon.ld1x2.v8i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int16x4x2_t @llvm.aarch64.neon.ld1x2.v4i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int32x2x2_t @llvm.aarch64.neon.ld1x2.v2i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld1x2.v2f32.p0f32(float*) nounwind readonly
declare %struct.__neon_int64x1x2_t @llvm.aarch64.neon.ld1x2.v1i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_float64x1x2_t @llvm.aarch64.neon.ld1x2.v1f64.p0f64(double*) nounwind readonly

define %struct.__neon_int8x8x2_t @ld1_x2_v8i8(i8* %addr) {
; CHECK-LABEL: ld1_x2_v8i8:
; CHECK: ld1.8b { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int8x8x2_t @llvm.aarch64.neon.ld1x2.v8i8.p0i8(i8* %addr)
  ret %struct.__neon_int8x8x2_t %val
}

define %struct.__neon_int16x4x2_t @ld1_x2_v4i16(i16* %addr) {
; CHECK-LABEL: ld1_x2_v4i16:
; CHECK: ld1.4h { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int16x4x2_t @llvm.aarch64.neon.ld1x2.v4i16.p0i16(i16* %addr)
  ret %struct.__neon_int16x4x2_t %val
}

define %struct.__neon_int32x2x2_t @ld1_x2_v2i32(i32* %addr) {
; CHECK-LABEL: ld1_x2_v2i32:
; CHECK: ld1.2s { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int32x2x2_t @llvm.aarch64.neon.ld1x2.v2i32.p0i32(i32* %addr)
  ret %struct.__neon_int32x2x2_t %val
}

define %struct.__neon_float32x2x2_t @ld1_x2_v2f32(float* %addr) {
; CHECK-LABEL: ld1_x2_v2f32:
; CHECK: ld1.2s { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld1x2.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x2_t %val
}

define %struct.__neon_int64x1x2_t @ld1_x2_v1i64(i64* %addr) {
; CHECK-LABEL: ld1_x2_v1i64:
; CHECK: ld1.1d { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int64x1x2_t @llvm.aarch64.neon.ld1x2.v1i64.p0i64(i64* %addr)
  ret %struct.__neon_int64x1x2_t %val
}

define %struct.__neon_float64x1x2_t @ld1_x2_v1f64(double* %addr) {
; CHECK-LABEL: ld1_x2_v1f64:
; CHECK: ld1.1d { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float64x1x2_t @llvm.aarch64.neon.ld1x2.v1f64.p0f64(double* %addr)
  ret %struct.__neon_float64x1x2_t %val
}


%struct.__neon_float32x4x2_t = type { <4 x float>,  <4 x float> }
%struct.__neon_float32x4x3_t = type { <4 x float>,  <4 x float>,  <4 x float> }
%struct.__neon_float32x4x4_t = type { <4 x float>,  <4 x float>, <4 x float>,  <4 x float> }

%struct.__neon_float64x2x2_t = type { <2 x double>,  <2 x double> }
%struct.__neon_float64x2x3_t = type { <2 x double>,  <2 x double>,  <2 x double> }
%struct.__neon_float64x2x4_t = type { <2 x double>,  <2 x double>, <2 x double>,  <2 x double> }

declare %struct.__neon_int8x16x2_t @llvm.aarch64.neon.ld1x2.v16i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int16x8x2_t @llvm.aarch64.neon.ld1x2.v8i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int32x4x2_t @llvm.aarch64.neon.ld1x2.v4i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_float32x4x2_t @llvm.aarch64.neon.ld1x2.v4f32.p0f32(float*) nounwind readonly
declare %struct.__neon_int64x2x2_t @llvm.aarch64.neon.ld1x2.v2i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_float64x2x2_t @llvm.aarch64.neon.ld1x2.v2f64.p0f64(double*) nounwind readonly

define %struct.__neon_int8x16x2_t @ld1_x2_v16i8(i8* %addr) {
; CHECK-LABEL: ld1_x2_v16i8:
; CHECK: ld1.16b { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int8x16x2_t @llvm.aarch64.neon.ld1x2.v16i8.p0i8(i8* %addr)
  ret %struct.__neon_int8x16x2_t %val
}

define %struct.__neon_int16x8x2_t @ld1_x2_v8i16(i16* %addr) {
; CHECK-LABEL: ld1_x2_v8i16:
; CHECK: ld1.8h { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int16x8x2_t @llvm.aarch64.neon.ld1x2.v8i16.p0i16(i16* %addr)
  ret %struct.__neon_int16x8x2_t %val
}

define %struct.__neon_int32x4x2_t @ld1_x2_v4i32(i32* %addr) {
; CHECK-LABEL: ld1_x2_v4i32:
; CHECK: ld1.4s { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int32x4x2_t @llvm.aarch64.neon.ld1x2.v4i32.p0i32(i32* %addr)
  ret %struct.__neon_int32x4x2_t %val
}

define %struct.__neon_float32x4x2_t @ld1_x2_v4f32(float* %addr) {
; CHECK-LABEL: ld1_x2_v4f32:
; CHECK: ld1.4s { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float32x4x2_t @llvm.aarch64.neon.ld1x2.v4f32.p0f32(float* %addr)
  ret %struct.__neon_float32x4x2_t %val
}

define %struct.__neon_int64x2x2_t @ld1_x2_v2i64(i64* %addr) {
; CHECK-LABEL: ld1_x2_v2i64:
; CHECK: ld1.2d { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int64x2x2_t @llvm.aarch64.neon.ld1x2.v2i64.p0i64(i64* %addr)
  ret %struct.__neon_int64x2x2_t %val
}

define %struct.__neon_float64x2x2_t @ld1_x2_v2f64(double* %addr) {
; CHECK-LABEL: ld1_x2_v2f64:
; CHECK: ld1.2d { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float64x2x2_t @llvm.aarch64.neon.ld1x2.v2f64.p0f64(double* %addr)
  ret %struct.__neon_float64x2x2_t %val
}

declare %struct.__neon_int8x8x3_t @llvm.aarch64.neon.ld1x3.v8i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.aarch64.neon.ld1x3.v4i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.aarch64.neon.ld1x3.v2i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld1x3.v2f32.p0f32(float*) nounwind readonly
declare %struct.__neon_int64x1x3_t @llvm.aarch64.neon.ld1x3.v1i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_float64x1x3_t @llvm.aarch64.neon.ld1x3.v1f64.p0f64(double*) nounwind readonly

define %struct.__neon_int8x8x3_t @ld1_x3_v8i8(i8* %addr) {
; CHECK-LABEL: ld1_x3_v8i8:
; CHECK: ld1.8b { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int8x8x3_t @llvm.aarch64.neon.ld1x3.v8i8.p0i8(i8* %addr)
  ret %struct.__neon_int8x8x3_t %val
}

define %struct.__neon_int16x4x3_t @ld1_x3_v4i16(i16* %addr) {
; CHECK-LABEL: ld1_x3_v4i16:
; CHECK: ld1.4h { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int16x4x3_t @llvm.aarch64.neon.ld1x3.v4i16.p0i16(i16* %addr)
  ret %struct.__neon_int16x4x3_t %val
}

define %struct.__neon_int32x2x3_t @ld1_x3_v2i32(i32* %addr) {
; CHECK-LABEL: ld1_x3_v2i32:
; CHECK: ld1.2s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int32x2x3_t @llvm.aarch64.neon.ld1x3.v2i32.p0i32(i32* %addr)
  ret %struct.__neon_int32x2x3_t %val
}

define %struct.__neon_float32x2x3_t @ld1_x3_v2f32(float* %addr) {
; CHECK-LABEL: ld1_x3_v2f32:
; CHECK: ld1.2s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld1x3.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x3_t %val
}

define %struct.__neon_int64x1x3_t @ld1_x3_v1i64(i64* %addr) {
; CHECK-LABEL: ld1_x3_v1i64:
; CHECK: ld1.1d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int64x1x3_t @llvm.aarch64.neon.ld1x3.v1i64.p0i64(i64* %addr)
  ret %struct.__neon_int64x1x3_t %val
}

define %struct.__neon_float64x1x3_t @ld1_x3_v1f64(double* %addr) {
; CHECK-LABEL: ld1_x3_v1f64:
; CHECK: ld1.1d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float64x1x3_t @llvm.aarch64.neon.ld1x3.v1f64.p0f64(double* %addr)
  ret %struct.__neon_float64x1x3_t %val
}

declare %struct.__neon_int8x16x3_t @llvm.aarch64.neon.ld1x3.v16i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int16x8x3_t @llvm.aarch64.neon.ld1x3.v8i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.aarch64.neon.ld1x3.v4i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_float32x4x3_t @llvm.aarch64.neon.ld1x3.v4f32.p0f32(float*) nounwind readonly
declare %struct.__neon_int64x2x3_t @llvm.aarch64.neon.ld1x3.v2i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_float64x2x3_t @llvm.aarch64.neon.ld1x3.v2f64.p0f64(double*) nounwind readonly

define %struct.__neon_int8x16x3_t @ld1_x3_v16i8(i8* %addr) {
; CHECK-LABEL: ld1_x3_v16i8:
; CHECK: ld1.16b { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int8x16x3_t @llvm.aarch64.neon.ld1x3.v16i8.p0i8(i8* %addr)
  ret %struct.__neon_int8x16x3_t %val
}

define %struct.__neon_int16x8x3_t @ld1_x3_v8i16(i16* %addr) {
; CHECK-LABEL: ld1_x3_v8i16:
; CHECK: ld1.8h { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int16x8x3_t @llvm.aarch64.neon.ld1x3.v8i16.p0i16(i16* %addr)
  ret %struct.__neon_int16x8x3_t %val
}

define %struct.__neon_int32x4x3_t @ld1_x3_v4i32(i32* %addr) {
; CHECK-LABEL: ld1_x3_v4i32:
; CHECK: ld1.4s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int32x4x3_t @llvm.aarch64.neon.ld1x3.v4i32.p0i32(i32* %addr)
  ret %struct.__neon_int32x4x3_t %val
}

define %struct.__neon_float32x4x3_t @ld1_x3_v4f32(float* %addr) {
; CHECK-LABEL: ld1_x3_v4f32:
; CHECK: ld1.4s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float32x4x3_t @llvm.aarch64.neon.ld1x3.v4f32.p0f32(float* %addr)
  ret %struct.__neon_float32x4x3_t %val
}

define %struct.__neon_int64x2x3_t @ld1_x3_v2i64(i64* %addr) {
; CHECK-LABEL: ld1_x3_v2i64:
; CHECK: ld1.2d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int64x2x3_t @llvm.aarch64.neon.ld1x3.v2i64.p0i64(i64* %addr)
  ret %struct.__neon_int64x2x3_t %val
}

define %struct.__neon_float64x2x3_t @ld1_x3_v2f64(double* %addr) {
; CHECK-LABEL: ld1_x3_v2f64:
; CHECK: ld1.2d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float64x2x3_t @llvm.aarch64.neon.ld1x3.v2f64.p0f64(double* %addr)
  ret %struct.__neon_float64x2x3_t %val
}

declare %struct.__neon_int8x8x4_t @llvm.aarch64.neon.ld1x4.v8i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int16x4x4_t @llvm.aarch64.neon.ld1x4.v4i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int32x2x4_t @llvm.aarch64.neon.ld1x4.v2i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld1x4.v2f32.p0f32(float*) nounwind readonly
declare %struct.__neon_int64x1x4_t @llvm.aarch64.neon.ld1x4.v1i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_float64x1x4_t @llvm.aarch64.neon.ld1x4.v1f64.p0f64(double*) nounwind readonly

define %struct.__neon_int8x8x4_t @ld1_x4_v8i8(i8* %addr) {
; CHECK-LABEL: ld1_x4_v8i8:
; CHECK: ld1.8b { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int8x8x4_t @llvm.aarch64.neon.ld1x4.v8i8.p0i8(i8* %addr)
  ret %struct.__neon_int8x8x4_t %val
}

define %struct.__neon_int16x4x4_t @ld1_x4_v4i16(i16* %addr) {
; CHECK-LABEL: ld1_x4_v4i16:
; CHECK: ld1.4h { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int16x4x4_t @llvm.aarch64.neon.ld1x4.v4i16.p0i16(i16* %addr)
  ret %struct.__neon_int16x4x4_t %val
}

define %struct.__neon_int32x2x4_t @ld1_x4_v2i32(i32* %addr) {
; CHECK-LABEL: ld1_x4_v2i32:
; CHECK: ld1.2s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int32x2x4_t @llvm.aarch64.neon.ld1x4.v2i32.p0i32(i32* %addr)
  ret %struct.__neon_int32x2x4_t %val
}

define %struct.__neon_float32x2x4_t @ld1_x4_v2f32(float* %addr) {
; CHECK-LABEL: ld1_x4_v2f32:
; CHECK: ld1.2s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld1x4.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x4_t %val
}

define %struct.__neon_int64x1x4_t @ld1_x4_v1i64(i64* %addr) {
; CHECK-LABEL: ld1_x4_v1i64:
; CHECK: ld1.1d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int64x1x4_t @llvm.aarch64.neon.ld1x4.v1i64.p0i64(i64* %addr)
  ret %struct.__neon_int64x1x4_t %val
}

define %struct.__neon_float64x1x4_t @ld1_x4_v1f64(double* %addr) {
; CHECK-LABEL: ld1_x4_v1f64:
; CHECK: ld1.1d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float64x1x4_t @llvm.aarch64.neon.ld1x4.v1f64.p0f64(double* %addr)
  ret %struct.__neon_float64x1x4_t %val
}

declare %struct.__neon_int8x16x4_t @llvm.aarch64.neon.ld1x4.v16i8.p0i8(i8*) nounwind readonly
declare %struct.__neon_int16x8x4_t @llvm.aarch64.neon.ld1x4.v8i16.p0i16(i16*) nounwind readonly
declare %struct.__neon_int32x4x4_t @llvm.aarch64.neon.ld1x4.v4i32.p0i32(i32*) nounwind readonly
declare %struct.__neon_float32x4x4_t @llvm.aarch64.neon.ld1x4.v4f32.p0f32(float*) nounwind readonly
declare %struct.__neon_int64x2x4_t @llvm.aarch64.neon.ld1x4.v2i64.p0i64(i64*) nounwind readonly
declare %struct.__neon_float64x2x4_t @llvm.aarch64.neon.ld1x4.v2f64.p0f64(double*) nounwind readonly

define %struct.__neon_int8x16x4_t @ld1_x4_v16i8(i8* %addr) {
; CHECK-LABEL: ld1_x4_v16i8:
; CHECK: ld1.16b { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int8x16x4_t @llvm.aarch64.neon.ld1x4.v16i8.p0i8(i8* %addr)
  ret %struct.__neon_int8x16x4_t %val
}

define %struct.__neon_int16x8x4_t @ld1_x4_v8i16(i16* %addr) {
; CHECK-LABEL: ld1_x4_v8i16:
; CHECK: ld1.8h { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int16x8x4_t @llvm.aarch64.neon.ld1x4.v8i16.p0i16(i16* %addr)
  ret %struct.__neon_int16x8x4_t %val
}

define %struct.__neon_int32x4x4_t @ld1_x4_v4i32(i32* %addr) {
; CHECK-LABEL: ld1_x4_v4i32:
; CHECK: ld1.4s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int32x4x4_t @llvm.aarch64.neon.ld1x4.v4i32.p0i32(i32* %addr)
  ret %struct.__neon_int32x4x4_t %val
}

define %struct.__neon_float32x4x4_t @ld1_x4_v4f32(float* %addr) {
; CHECK-LABEL: ld1_x4_v4f32:
; CHECK: ld1.4s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float32x4x4_t @llvm.aarch64.neon.ld1x4.v4f32.p0f32(float* %addr)
  ret %struct.__neon_float32x4x4_t %val
}

define %struct.__neon_int64x2x4_t @ld1_x4_v2i64(i64* %addr) {
; CHECK-LABEL: ld1_x4_v2i64:
; CHECK: ld1.2d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_int64x2x4_t @llvm.aarch64.neon.ld1x4.v2i64.p0i64(i64* %addr)
  ret %struct.__neon_int64x2x4_t %val
}

define %struct.__neon_float64x2x4_t @ld1_x4_v2f64(double* %addr) {
; CHECK-LABEL: ld1_x4_v2f64:
; CHECK: ld1.2d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  %val = call %struct.__neon_float64x2x4_t @llvm.aarch64.neon.ld1x4.v2f64.p0f64(double* %addr)
  ret %struct.__neon_float64x2x4_t %val
}
