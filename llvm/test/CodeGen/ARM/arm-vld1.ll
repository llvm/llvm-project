; RUN: llc < %s -mtriple=armv8-linux-gnueabi -verify-machineinstrs \
; RUN:     -asm-verbose=false | FileCheck %s

%struct.uint16x4x2_t = type { <4 x i16>, <4 x i16> }
%struct.uint16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.uint16x4x4_t = type { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }

%struct.uint32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.uint32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.uint32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }

%struct.uint64x1x2_t = type { <1 x i64>, <1 x i64> }
%struct.uint64x1x3_t = type { <1 x i64>, <1 x i64>, <1 x i64> }
%struct.uint64x1x4_t = type { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }

%struct.uint8x8x2_t = type { <8 x i8>, <8 x i8> }
%struct.uint8x8x3_t = type { <8 x i8>, <8 x i8>, <8 x i8> }
%struct.uint8x8x4_t = type { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }

%struct.uint16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.uint16x8x3_t = type { <8 x i16>, <8 x i16>, <8 x i16> }
%struct.uint16x8x4_t = type { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }

%struct.uint32x4x2_t = type { <4 x i32>, <4 x i32> }
%struct.uint32x4x3_t = type { <4 x i32>, <4 x i32>, <4 x i32> }
%struct.uint32x4x4_t = type { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }

%struct.uint64x2x2_t = type { <2 x i64>, <2 x i64> }
%struct.uint64x2x3_t = type { <2 x i64>, <2 x i64>, <2 x i64> }
%struct.uint64x2x4_t = type { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }

%struct.uint8x16x2_t = type { <16 x i8>, <16 x i8> }
%struct.uint8x16x3_t = type { <16 x i8>, <16 x i8>, <16 x i8> }
%struct.uint8x16x4_t = type { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }

declare %struct.uint16x4x2_t @llvm.arm.neon.vld1x2.v4i16.p0i16(i16*) nounwind readonly
declare %struct.uint16x4x3_t @llvm.arm.neon.vld1x3.v4i16.p0i16(i16*) nounwind readonly
declare %struct.uint16x4x4_t @llvm.arm.neon.vld1x4.v4i16.p0i16(i16*) nounwind readonly

declare %struct.uint32x2x2_t @llvm.arm.neon.vld1x2.v2i32.p0i32(i32*) nounwind readonly
declare %struct.uint32x2x3_t @llvm.arm.neon.vld1x3.v2i32.p0i32(i32*) nounwind readonly
declare %struct.uint32x2x4_t @llvm.arm.neon.vld1x4.v2i32.p0i32(i32*) nounwind readonly

declare %struct.uint64x1x2_t @llvm.arm.neon.vld1x2.v1i64.p0i64(i64*) nounwind readonly
declare %struct.uint64x1x3_t @llvm.arm.neon.vld1x3.v1i64.p0i64(i64*) nounwind readonly
declare %struct.uint64x1x4_t @llvm.arm.neon.vld1x4.v1i64.p0i64(i64*) nounwind readonly

declare %struct.uint8x8x2_t @llvm.arm.neon.vld1x2.v8i8.p0i8(i8*) nounwind readonly
declare %struct.uint8x8x3_t @llvm.arm.neon.vld1x3.v8i8.p0i8(i8*) nounwind readonly
declare %struct.uint8x8x4_t @llvm.arm.neon.vld1x4.v8i8.p0i8(i8*) nounwind readonly

declare %struct.uint16x8x2_t @llvm.arm.neon.vld1x2.v8i16.p0i16(i16*) nounwind readonly
declare %struct.uint16x8x3_t @llvm.arm.neon.vld1x3.v8i16.p0i16(i16*) nounwind readonly
declare %struct.uint16x8x4_t @llvm.arm.neon.vld1x4.v8i16.p0i16(i16*) nounwind readonly

declare %struct.uint32x4x2_t @llvm.arm.neon.vld1x2.v4i32.p0i32(i32*) nounwind readonly
declare %struct.uint32x4x3_t @llvm.arm.neon.vld1x3.v4i32.p0i32(i32*) nounwind readonly
declare %struct.uint32x4x4_t @llvm.arm.neon.vld1x4.v4i32.p0i32(i32*) nounwind readonly

declare %struct.uint64x2x2_t @llvm.arm.neon.vld1x2.v2i64.p0i64(i64*) nounwind readonly
declare %struct.uint64x2x3_t @llvm.arm.neon.vld1x3.v2i64.p0i64(i64*) nounwind readonly
declare %struct.uint64x2x4_t @llvm.arm.neon.vld1x4.v2i64.p0i64(i64*) nounwind readonly

declare %struct.uint8x16x2_t @llvm.arm.neon.vld1x2.v16i8.p0i8(i8*) nounwind readonly
declare %struct.uint8x16x3_t @llvm.arm.neon.vld1x3.v16i8.p0i8(i8*) nounwind readonly
declare %struct.uint8x16x4_t @llvm.arm.neon.vld1x4.v16i8.p0i8(i8*) nounwind readonly

; CHECK-LABEL: test_vld1_u16_x2
; CHECK: vld1.16 {d16, d17}, [r0:64]
define %struct.uint16x4x2_t @test_vld1_u16_x2(i16* %a) nounwind {
  %tmp = tail call %struct.uint16x4x2_t @llvm.arm.neon.vld1x2.v4i16.p0i16(i16* %a)
  ret %struct.uint16x4x2_t %tmp
}

; CHECK-LABEL: test_vld1_u16_x3
; CHECK: vld1.16 {d16, d17, d18}, [r1:64]
define %struct.uint16x4x3_t @test_vld1_u16_x3(i16* %a) nounwind {
  %tmp = tail call %struct.uint16x4x3_t @llvm.arm.neon.vld1x3.v4i16.p0i16(i16* %a)
  ret %struct.uint16x4x3_t %tmp
}

; CHECK-LABEL: test_vld1_u16_x4
; CHECK: vld1.16 {d16, d17, d18, d19}, [r1:256]
define %struct.uint16x4x4_t @test_vld1_u16_x4(i16* %a) nounwind {
  %tmp = tail call %struct.uint16x4x4_t @llvm.arm.neon.vld1x4.v4i16.p0i16(i16* %a)
  ret %struct.uint16x4x4_t %tmp
}

; CHECK-LABEL: test_vld1_u32_x2
; CHECK: vld1.32 {d16, d17}, [r0:64]
define %struct.uint32x2x2_t @test_vld1_u32_x2(i32* %a) nounwind {
  %tmp = tail call %struct.uint32x2x2_t @llvm.arm.neon.vld1x2.v2i32.p0i32(i32* %a)
  ret %struct.uint32x2x2_t %tmp
}

; CHECK-LABEL: test_vld1_u32_x3
; CHECK: vld1.32 {d16, d17, d18}, [r1:64]
define %struct.uint32x2x3_t @test_vld1_u32_x3(i32* %a) nounwind {
  %tmp = tail call %struct.uint32x2x3_t @llvm.arm.neon.vld1x3.v2i32.p0i32(i32* %a)
  ret %struct.uint32x2x3_t %tmp
}

; CHECK-LABEL: test_vld1_u32_x4
; CHECK: vld1.32 {d16, d17, d18, d19}, [r1:256]
define %struct.uint32x2x4_t @test_vld1_u32_x4(i32* %a) nounwind {
  %tmp = tail call %struct.uint32x2x4_t @llvm.arm.neon.vld1x4.v2i32.p0i32(i32* %a)
  ret %struct.uint32x2x4_t %tmp
}

; CHECK-LABEL: test_vld1_u64_x2
; CHECK: vld1.64 {d16, d17}, [r0:64]
define %struct.uint64x1x2_t @test_vld1_u64_x2(i64* %a) nounwind {
  %tmp = tail call %struct.uint64x1x2_t @llvm.arm.neon.vld1x2.v1i64.p0i64(i64* %a)
  ret %struct.uint64x1x2_t %tmp
}

; CHECK-LABEL: test_vld1_u64_x3
; CHECK: vld1.64 {d16, d17, d18}, [r1:64]
define %struct.uint64x1x3_t @test_vld1_u64_x3(i64* %a) nounwind {
  %tmp = tail call %struct.uint64x1x3_t @llvm.arm.neon.vld1x3.v1i64.p0i64(i64* %a)
  ret %struct.uint64x1x3_t %tmp
}

; CHECK-LABEL: test_vld1_u64_x4
; CHECK: vld1.64 {d16, d17, d18, d19}, [r1:256]
define %struct.uint64x1x4_t @test_vld1_u64_x4(i64* %a) nounwind {
  %tmp = tail call %struct.uint64x1x4_t @llvm.arm.neon.vld1x4.v1i64.p0i64(i64* %a)
  ret %struct.uint64x1x4_t %tmp
}

; CHECK-LABEL: test_vld1_u8_x2
; CHECK: vld1.8 {d16, d17}, [r0:64]
define %struct.uint8x8x2_t @test_vld1_u8_x2(i8* %a) nounwind {
  %tmp = tail call %struct.uint8x8x2_t @llvm.arm.neon.vld1x2.v8i8.p0i8(i8* %a)
  ret %struct.uint8x8x2_t %tmp
}

; CHECK-LABEL: test_vld1_u8_x3
; CHECK: vld1.8 {d16, d17, d18}, [r1:64]
define %struct.uint8x8x3_t @test_vld1_u8_x3(i8* %a) nounwind {
  %tmp = tail call %struct.uint8x8x3_t @llvm.arm.neon.vld1x3.v8i8.p0i8(i8* %a)
  ret %struct.uint8x8x3_t %tmp
}

; CHECK-LABEL: test_vld1_u8_x4
; CHECK: vld1.8 {d16, d17, d18, d19}, [r1:256]
define %struct.uint8x8x4_t @test_vld1_u8_x4(i8* %a) nounwind {
  %tmp = tail call %struct.uint8x8x4_t @llvm.arm.neon.vld1x4.v8i8.p0i8(i8* %a)
  ret %struct.uint8x8x4_t %tmp
}

; CHECK-LABEL: test_vld1q_u16_x2
; CHECK: vld1.16 {d16, d17, d18, d19}, [r1:256]
define %struct.uint16x8x2_t @test_vld1q_u16_x2(i16* %a) nounwind {
  %tmp = tail call %struct.uint16x8x2_t @llvm.arm.neon.vld1x2.v8i16.p0i16(i16* %a)
  ret %struct.uint16x8x2_t %tmp
}

; CHECK-LABEL: test_vld1q_u16_x3
; CHECK: vld1.16 {d16, d17, d18}, [r1:64]!
; CHECK: vld1.16 {d19, d20, d21}, [r1:64]
define %struct.uint16x8x3_t @test_vld1q_u16_x3(i16* %a) nounwind {
  %tmp = tail call %struct.uint16x8x3_t @llvm.arm.neon.vld1x3.v8i16.p0i16(i16* %a)
  ret %struct.uint16x8x3_t %tmp
}

; CHECK-LABEL: test_vld1q_u16_x4
; CHECK: vld1.16 {d16, d17, d18, d19}, [r1:256]!
; CHECK: vld1.16 {d20, d21, d22, d23}, [r1:256]
define %struct.uint16x8x4_t @test_vld1q_u16_x4(i16* %a) nounwind {
  %tmp = tail call %struct.uint16x8x4_t @llvm.arm.neon.vld1x4.v8i16.p0i16(i16* %a)
  ret %struct.uint16x8x4_t %tmp
}

; CHECK-LABEL: test_vld1q_u32_x2
; CHECK: vld1.32 {d16, d17, d18, d19}, [r1:256]
define %struct.uint32x4x2_t @test_vld1q_u32_x2(i32* %a) nounwind {
  %tmp = tail call %struct.uint32x4x2_t @llvm.arm.neon.vld1x2.v4i32.p0i32(i32* %a)
  ret %struct.uint32x4x2_t %tmp
}

; CHECK-LABEL: test_vld1q_u32_x3
; CHECK: vld1.32 {d16, d17, d18}, [r1:64]!
; CHECK: vld1.32 {d19, d20, d21}, [r1:64]
define %struct.uint32x4x3_t @test_vld1q_u32_x3(i32* %a) nounwind {
  %tmp = tail call %struct.uint32x4x3_t @llvm.arm.neon.vld1x3.v4i32.p0i32(i32* %a)
  ret %struct.uint32x4x3_t %tmp
}

; CHECK-LABEL: test_vld1q_u32_x4
; CHECK: vld1.32 {d16, d17, d18, d19}, [r1:256]!
; CHECK: vld1.32 {d20, d21, d22, d23}, [r1:256]
define %struct.uint32x4x4_t @test_vld1q_u32_x4(i32* %a) nounwind {
  %tmp = tail call %struct.uint32x4x4_t @llvm.arm.neon.vld1x4.v4i32.p0i32(i32* %a)
  ret %struct.uint32x4x4_t %tmp
}

; CHECK-LABEL: test_vld1q_u64_x2
; CHECK: vld1.64 {d16, d17, d18, d19}, [r1:256]
define %struct.uint64x2x2_t @test_vld1q_u64_x2(i64* %a) nounwind {
  %tmp = tail call %struct.uint64x2x2_t @llvm.arm.neon.vld1x2.v2i64.p0i64(i64* %a)
  ret %struct.uint64x2x2_t %tmp
}

; CHECK-LABEL: test_vld1q_u64_x3
; CHECK: vld1.64 {d16, d17, d18}, [r1:64]!
; CHECK: vld1.64 {d19, d20, d21}, [r1:64]
define %struct.uint64x2x3_t @test_vld1q_u64_x3(i64* %a) nounwind {
  %tmp = tail call %struct.uint64x2x3_t @llvm.arm.neon.vld1x3.v2i64.p0i64(i64* %a)
  ret %struct.uint64x2x3_t %tmp
}

; CHECK-LABEL: test_vld1q_u64_x4
; CHECK: vld1.64 {d16, d17, d18, d19}, [r1:256]!
; CHECK: vld1.64 {d20, d21, d22, d23}, [r1:256]
define %struct.uint64x2x4_t @test_vld1q_u64_x4(i64* %a) nounwind {
  %tmp = tail call %struct.uint64x2x4_t @llvm.arm.neon.vld1x4.v2i64.p0i64(i64* %a)
  ret %struct.uint64x2x4_t %tmp
}

; CHECK-LABEL: test_vld1q_u8_x2
; CHECK: vld1.8 {d16, d17, d18, d19}, [r1:256]
define %struct.uint8x16x2_t @test_vld1q_u8_x2(i8* %a) nounwind {
  %tmp = tail call %struct.uint8x16x2_t @llvm.arm.neon.vld1x2.v16i8.p0i8(i8* %a)
  ret %struct.uint8x16x2_t %tmp
}

; CHECK-LABEL: test_vld1q_u8_x3
; CHECK: vld1.8 {d16, d17, d18}, [r1:64]!
; CHECK: vld1.8 {d19, d20, d21}, [r1:64]
define %struct.uint8x16x3_t @test_vld1q_u8_x3(i8* %a) nounwind {
  %tmp = tail call %struct.uint8x16x3_t @llvm.arm.neon.vld1x3.v16i8.p0i8(i8* %a)
  ret %struct.uint8x16x3_t %tmp
}

; CHECK-LABEL: test_vld1q_u8_x4
; CHECK: vld1.8 {d16, d17, d18, d19}, [r1:256]!
; CHECK: vld1.8 {d20, d21, d22, d23}, [r1:256]
define %struct.uint8x16x4_t @test_vld1q_u8_x4(i8* %a) nounwind {
  %tmp = tail call %struct.uint8x16x4_t @llvm.arm.neon.vld1x4.v16i8.p0i8(i8* %a)
  ret %struct.uint8x16x4_t %tmp
}
