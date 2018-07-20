; RUN: llc < %s -mtriple=armv8-linux-gnueabi -verify-machineinstrs \
; RUN:     -asm-verbose=false | FileCheck %s

; %struct.uint16x4x2_t = type { <4 x i16>, <4 x i16> }
; %struct.uint16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
; %struct.uint16x4x4_t = type { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }

; %struct.uint32x2x2_t = type { <2 x i32>, <2 x i32> }
; %struct.uint32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
; %struct.uint32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }

; %struct.uint64x1x2_t = type { <1 x i64>, <1 x i64> }
; %struct.uint64x1x3_t = type { <1 x i64>, <1 x i64>, <1 x i64> }
; %struct.uint64x1x4_t = type { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }

; %struct.uint8x8x2_t = type { <8 x i8>, <8 x i8> }
; %struct.uint8x8x3_t = type { <8 x i8>, <8 x i8>, <8 x i8> }
; %struct.uint8x8x4_t = type { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }

; %struct.uint16x8x2_t = type { <8 x i16>, <8 x i16> }
; %struct.uint16x8x3_t = type { <8 x i16>, <8 x i16>, <8 x i16> }
; %struct.uint16x8x4_t = type { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }

; %struct.uint32x4x2_t = type { <4 x i32>, <4 x i32> }
; %struct.uint32x4x3_t = type { <4 x i32>, <4 x i32>, <4 x i32> }
; %struct.uint32x4x4_t = type { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }

; %struct.uint64x2x2_t = type { <2 x i64>, <2 x i64> }
; %struct.uint64x2x3_t = type { <2 x i64>, <2 x i64>, <2 x i64> }
; %struct.uint64x2x4_t = type { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }

; %struct.uint8x16x2_t = type { <16 x i8>, <16 x i8> }
; %struct.uint8x16x3_t = type { <16 x i8>, <16 x i8>, <16 x i8> }
; %struct.uint8x16x4_t = type { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }

%struct.uint16x4x2_t = type { [2 x <4 x i16>] }
%struct.uint16x4x3_t = type { [3 x <4 x i16>] }
%struct.uint16x4x4_t = type { [4 x <4 x i16>] }
%struct.uint32x2x2_t = type { [2 x <2 x i32>] }
%struct.uint32x2x3_t = type { [3 x <2 x i32>] }
%struct.uint32x2x4_t = type { [4 x <2 x i32>] }
%struct.uint64x1x2_t = type { [2 x <1 x i64>] }
%struct.uint64x1x3_t = type { [3 x <1 x i64>] }
%struct.uint64x1x4_t = type { [4 x <1 x i64>] }
%struct.uint8x8x2_t = type { [2 x <8 x i8>] }
%struct.uint8x8x3_t = type { [3 x <8 x i8>] }
%struct.uint8x8x4_t = type { [4 x <8 x i8>] }
%struct.uint16x8x2_t = type { [2 x <8 x i16>] }
%struct.uint16x8x3_t = type { [3 x <8 x i16>] }
%struct.uint16x8x4_t = type { [4 x <8 x i16>] }
%struct.uint32x4x2_t = type { [2 x <4 x i32>] }
%struct.uint32x4x3_t = type { [3 x <4 x i32>] }
%struct.uint32x4x4_t = type { [4 x <4 x i32>] }
%struct.uint64x2x2_t = type { [2 x <2 x i64>] }
%struct.uint64x2x3_t = type { [3 x <2 x i64>] }
%struct.uint64x2x4_t = type { [4 x <2 x i64>] }
%struct.uint8x16x2_t = type { [2 x <16 x i8>] }
%struct.uint8x16x3_t = type { [3 x <16 x i8>] }
%struct.uint8x16x4_t = type { [4 x <16 x i8>] }

declare void @llvm.arm.neon.vst1x2.p0i16.v4i16(i16* nocapture, <4 x i16>, <4 x i16>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x3.p0i16.v4i16(i16* nocapture, <4 x i16>, <4 x i16>, <4 x i16>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x4.p0i16.v4i16(i16* nocapture, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>) argmemonly nounwind

declare void @llvm.arm.neon.vst1x2.p0i32.v2i32(i32* nocapture, <2 x i32>, <2 x i32>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x3.p0i32.v2i32(i32* nocapture, <2 x i32>, <2 x i32>, <2 x i32>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x4.p0i32.v2i32(i32* nocapture, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>) argmemonly nounwind

declare void @llvm.arm.neon.vst1x2.p0i64.v1i64(i64* nocapture, <1 x i64>, <1 x i64>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x3.p0i64.v1i64(i64* nocapture, <1 x i64>, <1 x i64>, <1 x i64>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x4.p0i64.v1i64(i64* nocapture, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>) argmemonly nounwind

declare void @llvm.arm.neon.vst1x2.p0i8.v8i8(i8* nocapture, <8 x i8>, <8 x i8>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x3.p0i8.v8i8(i8* nocapture, <8 x i8>, <8 x i8>, <8 x i8>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x4.p0i8.v8i8(i8* nocapture, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) argmemonly nounwind

declare void @llvm.arm.neon.vst1x2.p0i16.v8i16(i16* nocapture, <8 x i16>, <8 x i16>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x3.p0i16.v8i16(i16* nocapture, <8 x i16>, <8 x i16>, <8 x i16>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x4.p0i16.v8i16(i16* nocapture, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>) argmemonly nounwind

declare void @llvm.arm.neon.vst1x2.p0i32.v4i32(i32* nocapture, <4 x i32>, <4 x i32>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x3.p0i32.v4i32(i32* nocapture, <4 x i32>, <4 x i32>, <4 x i32>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x4.p0i32.v4i32(i32* nocapture, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>) argmemonly nounwind

declare void @llvm.arm.neon.vst1x2.p0i64.v2i64(i64* nocapture, <2 x i64>, <2 x i64>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x3.p0i64.v2i64(i64* nocapture, <2 x i64>, <2 x i64>, <2 x i64>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x4.p0i64.v2i64(i64* nocapture, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>) argmemonly nounwind

declare void @llvm.arm.neon.vst1x2.p0i8.v16i8(i8* nocapture, <16 x i8>, <16 x i8>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x3.p0i8.v16i8(i8* nocapture, <16 x i8>, <16 x i8>, <16 x i8>) argmemonly nounwind
declare void @llvm.arm.neon.vst1x4.p0i8.v16i8(i8* nocapture, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>) argmemonly nounwind

; CHECK-LABEL: test_vst1_u16_x2
; CHECK: vst1.16 {d16, d17}, [r0:64]
define void @test_vst1_u16_x2(i16* %a, %struct.uint16x4x2_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint16x4x2_t %b, 0, 0
  %b1 = extractvalue %struct.uint16x4x2_t %b, 0, 1
  tail call void @llvm.arm.neon.vst1x2.p0i16.v4i16(i16* %a, <4 x i16> %b0, <4 x i16> %b1)
  ret void
}

; CHECK-LABEL: test_vst1_u16_x3
; CHECK: vst1.16 {d16, d17, d18}, [r0:64]
define void @test_vst1_u16_x3(i16* %a, %struct.uint16x4x3_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint16x4x3_t %b, 0, 0
  %b1 = extractvalue %struct.uint16x4x3_t %b, 0, 1
  %b2 = extractvalue %struct.uint16x4x3_t %b, 0, 2
  tail call void @llvm.arm.neon.vst1x3.p0i16.v4i16(i16* %a, <4 x i16> %b0, <4 x i16> %b1, <4 x i16> %b2)
  ret void
}

; CHECK-LABEL: test_vst1_u16_x4
; CHECK: vst1.16 {d16, d17, d18, d19}, [r0:256]
define void @test_vst1_u16_x4(i16* %a, %struct.uint16x4x4_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint16x4x4_t %b, 0, 0
  %b1 = extractvalue %struct.uint16x4x4_t %b, 0, 1
  %b2 = extractvalue %struct.uint16x4x4_t %b, 0, 2
  %b3 = extractvalue %struct.uint16x4x4_t %b, 0, 3
  tail call void @llvm.arm.neon.vst1x4.p0i16.v4i16(i16* %a, <4 x i16> %b0, <4 x i16> %b1, <4 x i16> %b2, <4 x i16> %b3)
  ret void
}

; CHECK-LABEL: test_vst1_u32_x2
; CHECK: vst1.32 {d16, d17}, [r0:64]
define void @test_vst1_u32_x2(i32* %a, %struct.uint32x2x2_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint32x2x2_t %b, 0, 0
  %b1 = extractvalue %struct.uint32x2x2_t %b, 0, 1
  tail call void @llvm.arm.neon.vst1x2.p0i32.v2i32(i32* %a, <2 x i32> %b0, <2 x i32> %b1)
  ret void
}

; CHECK-LABEL: test_vst1_u32_x3
; CHECK: vst1.32 {d16, d17, d18}, [r0:64]
define void @test_vst1_u32_x3(i32* %a, %struct.uint32x2x3_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint32x2x3_t %b, 0, 0
  %b1 = extractvalue %struct.uint32x2x3_t %b, 0, 1
  %b2 = extractvalue %struct.uint32x2x3_t %b, 0, 2
  tail call void @llvm.arm.neon.vst1x3.p0i32.v2i32(i32* %a, <2 x i32> %b0, <2 x i32> %b1, <2 x i32> %b2)
  ret void
}

; CHECK-LABEL: test_vst1_u32_x4
; CHECK: vst1.32 {d16, d17, d18, d19}, [r0:256]
define void @test_vst1_u32_x4(i32* %a, %struct.uint32x2x4_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint32x2x4_t %b, 0, 0
  %b1 = extractvalue %struct.uint32x2x4_t %b, 0, 1
  %b2 = extractvalue %struct.uint32x2x4_t %b, 0, 2
  %b3 = extractvalue %struct.uint32x2x4_t %b, 0, 3
  tail call void @llvm.arm.neon.vst1x4.p0i32.v2i32(i32* %a, <2 x i32> %b0, <2 x i32> %b1, <2 x i32> %b2, <2 x i32> %b3)
  ret void
}

; CHECK-LABEL: test_vst1_u64_x2
; CHECK: vst1.64 {d16, d17}, [r0:64]
define void @test_vst1_u64_x2(i64* %a, %struct.uint64x1x2_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint64x1x2_t %b, 0, 0
  %b1 = extractvalue %struct.uint64x1x2_t %b, 0, 1
  tail call void @llvm.arm.neon.vst1x2.p0i64.v1i64(i64* %a, <1 x i64> %b0, <1 x i64> %b1)
  ret void
}

; CHECK-LABEL: test_vst1_u64_x3
; CHECK: vst1.64 {d16, d17, d18}, [r0:64]
define void @test_vst1_u64_x3(i64* %a, %struct.uint64x1x3_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint64x1x3_t %b, 0, 0
  %b1 = extractvalue %struct.uint64x1x3_t %b, 0, 1
  %b2 = extractvalue %struct.uint64x1x3_t %b, 0, 2
  tail call void @llvm.arm.neon.vst1x3.p0i64.v1i64(i64* %a, <1 x i64> %b0, <1 x i64> %b1, <1 x i64> %b2)
  ret void
}

; CHECK-LABEL: test_vst1_u64_x4
; CHECK: vst1.64 {d16, d17, d18, d19}, [r0:256]
define void @test_vst1_u64_x4(i64* %a, %struct.uint64x1x4_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint64x1x4_t %b, 0, 0
  %b1 = extractvalue %struct.uint64x1x4_t %b, 0, 1
  %b2 = extractvalue %struct.uint64x1x4_t %b, 0, 2
  %b3 = extractvalue %struct.uint64x1x4_t %b, 0, 3
  tail call void @llvm.arm.neon.vst1x4.p0i64.v1i64(i64* %a, <1 x i64> %b0, <1 x i64> %b1, <1 x i64> %b2, <1 x i64> %b3)
  ret void
}

; CHECK-LABEL: test_vst1_u8_x2
; CHECK: vst1.8 {d16, d17}, [r0:64]
define void @test_vst1_u8_x2(i8* %a, %struct.uint8x8x2_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint8x8x2_t %b, 0, 0
  %b1 = extractvalue %struct.uint8x8x2_t %b, 0, 1
  tail call void @llvm.arm.neon.vst1x2.p0i8.v8i8(i8* %a, <8 x i8> %b0, <8 x i8> %b1)
  ret void
}

; CHECK-LABEL: test_vst1_u8_x3
; CHECK: vst1.8 {d16, d17, d18}, [r0:64]
define void @test_vst1_u8_x3(i8* %a, %struct.uint8x8x3_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint8x8x3_t %b, 0, 0
  %b1 = extractvalue %struct.uint8x8x3_t %b, 0, 1
  %b2 = extractvalue %struct.uint8x8x3_t %b, 0, 2
  tail call void @llvm.arm.neon.vst1x3.p0i8.v8i8(i8* %a, <8 x i8> %b0, <8 x i8> %b1, <8 x i8> %b2)
  ret void
}

; CHECK-LABEL: test_vst1_u8_x4
; CHECK: vst1.8 {d16, d17, d18, d19}, [r0:256]
define void @test_vst1_u8_x4(i8* %a, %struct.uint8x8x4_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint8x8x4_t %b, 0, 0
  %b1 = extractvalue %struct.uint8x8x4_t %b, 0, 1
  %b2 = extractvalue %struct.uint8x8x4_t %b, 0, 2
  %b3 = extractvalue %struct.uint8x8x4_t %b, 0, 3
  tail call void @llvm.arm.neon.vst1x4.p0i8.v8i8(i8* %a, <8 x i8> %b0, <8 x i8> %b1, <8 x i8> %b2, <8 x i8> %b3)
  ret void
}

; CHECK-LABEL: test_vst1q_u16_x2
; CHECK: vst1.16 {d16, d17, d18, d19}, [r0:256]
define void @test_vst1q_u16_x2(i16* %a, %struct.uint16x8x2_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint16x8x2_t %b, 0, 0
  %b1 = extractvalue %struct.uint16x8x2_t %b, 0, 1
  tail call void @llvm.arm.neon.vst1x2.p0i16.v8i16(i16* %a, <8 x i16> %b0, <8 x i16> %b1)
  ret void
}

; CHECK-LABEL: test_vst1q_u16_x3
; CHECK: vst1.16 {d16, d17, d18}, [r0:64]!
; CHECK: vst1.16 {d19, d20, d21}, [r0:64]
define void @test_vst1q_u16_x3(i16* %a, %struct.uint16x8x3_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint16x8x3_t %b, 0, 0
  %b1 = extractvalue %struct.uint16x8x3_t %b, 0, 1
  %b2 = extractvalue %struct.uint16x8x3_t %b, 0, 2
  tail call void @llvm.arm.neon.vst1x3.p0i16.v8i16(i16* %a, <8 x i16> %b0, <8 x i16> %b1, <8 x i16> %b2)
  ret void
}

; CHECK-LABEL: test_vst1q_u16_x4
; CHECK: vst1.16 {d16, d17, d18, d19}, [r0:256]!
; CHECK: vst1.16 {d20, d21, d22, d23}, [r0:256]
define void @test_vst1q_u16_x4(i16* %a, %struct.uint16x8x4_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint16x8x4_t %b, 0, 0
  %b1 = extractvalue %struct.uint16x8x4_t %b, 0, 1
  %b2 = extractvalue %struct.uint16x8x4_t %b, 0, 2
  %b3 = extractvalue %struct.uint16x8x4_t %b, 0, 3
  tail call void @llvm.arm.neon.vst1x4.p0i16.v8i16(i16* %a, <8 x i16> %b0, <8 x i16> %b1, <8 x i16> %b2, <8 x i16> %b3)
  ret void
}

; CHECK-LABEL: test_vst1q_u32_x2
; CHECK: vst1.32 {d16, d17, d18, d19}, [r0:256]
define void @test_vst1q_u32_x2(i32* %a, %struct.uint32x4x2_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint32x4x2_t %b, 0, 0
  %b1 = extractvalue %struct.uint32x4x2_t %b, 0, 1
  tail call void @llvm.arm.neon.vst1x2.p0i32.v4i32(i32* %a, <4 x i32> %b0, <4 x i32> %b1)
  ret void
}

; CHECK-LABEL: test_vst1q_u32_x3
; CHECK: vst1.32 {d16, d17, d18}, [r0:64]!
; CHECK: vst1.32 {d19, d20, d21}, [r0:64]
define void @test_vst1q_u32_x3(i32* %a, %struct.uint32x4x3_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint32x4x3_t %b, 0, 0
  %b1 = extractvalue %struct.uint32x4x3_t %b, 0, 1
  %b2 = extractvalue %struct.uint32x4x3_t %b, 0, 2
  tail call void @llvm.arm.neon.vst1x3.p0i32.v4i32(i32* %a, <4 x i32> %b0, <4 x i32> %b1, <4 x i32> %b2)
  ret void
}

; CHECK-LABEL: test_vst1q_u32_x4
; CHECK: vst1.32 {d16, d17, d18, d19}, [r0:256]!
; CHECK: vst1.32 {d20, d21, d22, d23}, [r0:256]
define void @test_vst1q_u32_x4(i32* %a, %struct.uint32x4x4_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint32x4x4_t %b, 0, 0
  %b1 = extractvalue %struct.uint32x4x4_t %b, 0, 1
  %b2 = extractvalue %struct.uint32x4x4_t %b, 0, 2
  %b3 = extractvalue %struct.uint32x4x4_t %b, 0, 3
  tail call void @llvm.arm.neon.vst1x4.p0i32.v4i32(i32* %a, <4 x i32> %b0, <4 x i32> %b1, <4 x i32> %b2, <4 x i32> %b3)
  ret void
}

; CHECK-LABEL: test_vst1q_u64_x2
; CHECK: vst1.64 {d16, d17, d18, d19}, [r0:256]
define void @test_vst1q_u64_x2(i64* %a, %struct.uint64x2x2_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint64x2x2_t %b, 0, 0
  %b1 = extractvalue %struct.uint64x2x2_t %b, 0, 1
  tail call void @llvm.arm.neon.vst1x2.p0i64.v2i64(i64* %a, <2 x i64> %b0, <2 x i64> %b1)
  ret void
}

; CHECK-LABEL: test_vst1q_u64_x3
; CHECK: vst1.64 {d16, d17, d18}, [r0:64]!
; CHECK: vst1.64 {d19, d20, d21}, [r0:64]
define void @test_vst1q_u64_x3(i64* %a, %struct.uint64x2x3_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint64x2x3_t %b, 0, 0
  %b1 = extractvalue %struct.uint64x2x3_t %b, 0, 1
  %b2 = extractvalue %struct.uint64x2x3_t %b, 0, 2
  tail call void @llvm.arm.neon.vst1x3.p0i64.v2i64(i64* %a, <2 x i64> %b0, <2 x i64> %b1, <2 x i64> %b2)
  ret void
}

; CHECK-LABEL: test_vst1q_u64_x4
; CHECK: vst1.64 {d16, d17, d18, d19}, [r0:256]!
; CHECK: vst1.64 {d20, d21, d22, d23}, [r0:256]
define void @test_vst1q_u64_x4(i64* %a, %struct.uint64x2x4_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint64x2x4_t %b, 0, 0
  %b1 = extractvalue %struct.uint64x2x4_t %b, 0, 1
  %b2 = extractvalue %struct.uint64x2x4_t %b, 0, 2
  %b3 = extractvalue %struct.uint64x2x4_t %b, 0, 3
  tail call void @llvm.arm.neon.vst1x4.p0i64.v2i64(i64* %a, <2 x i64> %b0, <2 x i64> %b1, <2 x i64> %b2, <2 x i64> %b3)
  ret void
}

; CHECK-LABEL: test_vst1q_u8_x2
; CHECK: vst1.8 {d16, d17, d18, d19}, [r0:256]
define void @test_vst1q_u8_x2(i8* %a, %struct.uint8x16x2_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint8x16x2_t %b, 0, 0
  %b1 = extractvalue %struct.uint8x16x2_t %b, 0, 1
  tail call void @llvm.arm.neon.vst1x2.p0i8.v16i8(i8* %a, <16 x i8> %b0, <16 x i8> %b1)
  ret void
}

; CHECK-LABEL: test_vst1q_u8_x3
; CHECK: vst1.8 {d16, d17, d18}, [r0:64]!
; CHECK: vst1.8 {d19, d20, d21}, [r0:64]
define void @test_vst1q_u8_x3(i8* %a, %struct.uint8x16x3_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint8x16x3_t %b, 0, 0
  %b1 = extractvalue %struct.uint8x16x3_t %b, 0, 1
  %b2 = extractvalue %struct.uint8x16x3_t %b, 0, 2
  tail call void @llvm.arm.neon.vst1x3.p0i8.v16i8(i8* %a, <16 x i8> %b0, <16 x i8> %b1, <16 x i8> %b2)
  ret void
}

; CHECK-LABEL: test_vst1q_u8_x4
; CHECK: vst1.8 {d16, d17, d18, d19}, [r0:256]!
; CHECK: vst1.8 {d20, d21, d22, d23}, [r0:256]
define void @test_vst1q_u8_x4(i8* %a, %struct.uint8x16x4_t %b) nounwind {
entry:
  %b0 = extractvalue %struct.uint8x16x4_t %b, 0, 0
  %b1 = extractvalue %struct.uint8x16x4_t %b, 0, 1
  %b2 = extractvalue %struct.uint8x16x4_t %b, 0, 2
  %b3 = extractvalue %struct.uint8x16x4_t %b, 0, 3
  tail call void @llvm.arm.neon.vst1x4.p0i8.v16i8(i8* %a, <16 x i8> %b0, <16 x i8> %b1, <16 x i8> %b2, <16 x i8> %b3)
  ret void
}
