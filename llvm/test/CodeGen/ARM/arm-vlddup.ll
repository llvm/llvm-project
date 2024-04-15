; RUN: llc < %s -mtriple=armv8-linux-gnueabi --float-abi=hard -verify-machineinstrs \
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

%struct.uint8x16x2_t = type { <16 x i8>, <16 x i8> }
%struct.uint8x16x3_t = type { <16 x i8>, <16 x i8>, <16 x i8> }
%struct.uint8x16x4_t = type { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }

declare %struct.uint8x8x2_t @llvm.arm.neon.vld2dup.v8i8.p0(ptr, i32)
declare %struct.uint16x4x2_t @llvm.arm.neon.vld2dup.v4i16.p0(ptr, i32)
declare %struct.uint32x2x2_t @llvm.arm.neon.vld2dup.v2i32.p0(ptr, i32)
declare %struct.uint64x1x2_t @llvm.arm.neon.vld2dup.v1i64.p0(ptr, i32)

declare %struct.uint8x8x3_t @llvm.arm.neon.vld3dup.v8i8.p0(ptr, i32)
declare %struct.uint16x4x3_t @llvm.arm.neon.vld3dup.v4i16.p0(ptr, i32)
declare %struct.uint32x2x3_t @llvm.arm.neon.vld3dup.v2i32.p0(ptr, i32)
declare %struct.uint64x1x3_t @llvm.arm.neon.vld3dup.v1i64.p0(ptr, i32)

declare %struct.uint8x8x4_t @llvm.arm.neon.vld4dup.v8i8.p0(ptr, i32)
declare %struct.uint16x4x4_t @llvm.arm.neon.vld4dup.v4i16.p0(ptr, i32)
declare %struct.uint32x2x4_t @llvm.arm.neon.vld4dup.v2i32.p0(ptr, i32)
declare %struct.uint64x1x4_t @llvm.arm.neon.vld4dup.v1i64.p0(ptr, i32)

declare %struct.uint8x16x2_t @llvm.arm.neon.vld2dup.v16i8.p0(ptr, i32)
declare %struct.uint16x8x2_t @llvm.arm.neon.vld2dup.v8i16.p0(ptr, i32)
declare %struct.uint32x4x2_t @llvm.arm.neon.vld2dup.v4i32.p0(ptr, i32)

declare %struct.uint8x16x3_t @llvm.arm.neon.vld3dup.v16i8.p0(ptr, i32)
declare %struct.uint16x8x3_t @llvm.arm.neon.vld3dup.v8i16.p0(ptr, i32)
declare %struct.uint32x4x3_t @llvm.arm.neon.vld3dup.v4i32.p0(ptr, i32)

declare %struct.uint8x16x4_t @llvm.arm.neon.vld4dup.v16i8.p0(ptr, i32)
declare %struct.uint16x8x4_t @llvm.arm.neon.vld4dup.v8i16.p0(ptr, i32)
declare %struct.uint32x4x4_t @llvm.arm.neon.vld4dup.v4i32.p0(ptr, i32)

define %struct.uint16x4x2_t @test_vld2_dup_u16(ptr %src) {
; CHECK-LABEL: test_vld2_dup_u16:
; CHECK:         vld2.16 {d0[], d1[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint16x4x2_t @llvm.arm.neon.vld2dup.v4i16.p0(ptr %src, i32 2)
  ret %struct.uint16x4x2_t %tmp
}

define %struct.uint32x2x2_t @test_vld2_dup_u32(ptr %src) {
; CHECK-LABEL: test_vld2_dup_u32:
; CHECK:         vld2.32 {d0[], d1[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint32x2x2_t @llvm.arm.neon.vld2dup.v2i32.p0(ptr %src, i32 4)
  ret %struct.uint32x2x2_t %tmp
}

define %struct.uint64x1x2_t @test_vld2_dup_u64(ptr %src) {
; CHECK-LABEL: test_vld2_dup_u64:
; CHECK:         vld1.64 {d0, d1}, [r0:64]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint64x1x2_t @llvm.arm.neon.vld2dup.v1i64.p0(ptr %src, i32 8)
  ret %struct.uint64x1x2_t %tmp
}

define %struct.uint8x8x2_t @test_vld2_dup_u8(ptr %src) {
; CHECK-LABEL: test_vld2_dup_u8:
; CHECK:         vld2.8 {d0[], d1[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint8x8x2_t @llvm.arm.neon.vld2dup.v8i8.p0(ptr %src, i32 1)
  ret %struct.uint8x8x2_t %tmp
}

define %struct.uint16x4x3_t @test_vld3_dup_u16(ptr %src) {
; CHECK-LABEL: test_vld3_dup_u16:
; CHECK:         vld3.16 {d0[], d1[], d2[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint16x4x3_t @llvm.arm.neon.vld3dup.v4i16.p0(ptr %src, i32 2)
  ret %struct.uint16x4x3_t %tmp
}

define %struct.uint32x2x3_t @test_vld3_dup_u32(ptr %src) {
; CHECK-LABEL: test_vld3_dup_u32:
; CHECK:         vld3.32 {d0[], d1[], d2[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint32x2x3_t @llvm.arm.neon.vld3dup.v2i32.p0(ptr %src, i32 4)
  ret %struct.uint32x2x3_t %tmp
}

define %struct.uint64x1x3_t @test_vld3_dup_u64(ptr %src) {
; CHECK-LABEL: test_vld3_dup_u64:
; CHECK:         vld1.64 {d0, d1, d2}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint64x1x3_t @llvm.arm.neon.vld3dup.v1i64.p0(ptr %src, i32 8)
  ret %struct.uint64x1x3_t %tmp
}

define %struct.uint8x8x3_t @test_vld3_dup_u8(ptr %src) {
; CHECK-LABEL: test_vld3_dup_u8:
; CHECK:         vld3.8 {d0[], d1[], d2[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint8x8x3_t @llvm.arm.neon.vld3dup.v8i8.p0(ptr %src, i32 1)
  ret %struct.uint8x8x3_t %tmp
}

define %struct.uint16x4x4_t @test_vld4_dup_u16(ptr %src) {
; CHECK-LABEL: test_vld4_dup_u16:
; CHECK:         vld4.16 {d0[], d1[], d2[], d3[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint16x4x4_t @llvm.arm.neon.vld4dup.v4i16.p0(ptr %src, i32 2)
  ret %struct.uint16x4x4_t %tmp
}

define %struct.uint32x2x4_t @test_vld4_dup_u32(ptr %src) {
; CHECK-LABEL: test_vld4_dup_u32:
; CHECK:         vld4.32 {d0[], d1[], d2[], d3[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint32x2x4_t @llvm.arm.neon.vld4dup.v2i32.p0(ptr %src, i32 4)
  ret %struct.uint32x2x4_t %tmp
}

define %struct.uint64x1x4_t @test_vld4_dup_u64(ptr %src) {
; CHECK-LABEL: test_vld4_dup_u64:
; CHECK:         vld1.64 {d0, d1, d2, d3}, [r0:64]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint64x1x4_t @llvm.arm.neon.vld4dup.v1i64.p0(ptr %src, i32 8)
  ret %struct.uint64x1x4_t %tmp
}

define %struct.uint8x8x4_t @test_vld4_dup_u8(ptr %src) {
; CHECK-LABEL: test_vld4_dup_u8:
; CHECK:         vld4.8 {d0[], d1[], d2[], d3[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint8x8x4_t @llvm.arm.neon.vld4dup.v8i8.p0(ptr %src, i32 1)
  ret %struct.uint8x8x4_t %tmp
}

define %struct.uint16x8x2_t @test_vld2q_dup_u16(ptr %src) {
; CHECK-LABEL: test_vld2q_dup_u16:
; CHECK:         vld2.16 {d0[], d2[]}, [r0]
; CHECK-NEXT:    vld2.16 {d1[], d3[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint16x8x2_t @llvm.arm.neon.vld2dup.v8i16.p0(ptr %src, i32 2)
  ret %struct.uint16x8x2_t %tmp
}

define %struct.uint32x4x2_t @test_vld2q_dup_u32(ptr %src) {
; CHECK-LABEL: test_vld2q_dup_u32:
; CHECK:         vld2.32 {d0[], d2[]}, [r0]
; CHECK-NEXT:    vld2.32 {d1[], d3[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint32x4x2_t @llvm.arm.neon.vld2dup.v4i32.p0(ptr %src, i32 4)
  ret %struct.uint32x4x2_t %tmp
}

define %struct.uint8x16x2_t @test_vld2q_dup_u8(ptr %src) {
; CHECK-LABEL: test_vld2q_dup_u8:
; CHECK:         vld2.8 {d0[], d2[]}, [r0]
; CHECK-NEXT:    vld2.8 {d1[], d3[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint8x16x2_t @llvm.arm.neon.vld2dup.v16i8.p0(ptr %src, i32 1)
  ret %struct.uint8x16x2_t %tmp
}

define %struct.uint16x8x3_t @test_vld3q_dup_u16(ptr %src) {
; CHECK-LABEL: test_vld3q_dup_u16:
; CHECK:         vld3.16 {d0[], d2[], d4[]}, [r0]
; CHECK-NEXT:    vld3.16 {d1[], d3[], d5[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint16x8x3_t @llvm.arm.neon.vld3dup.v8i16.p0(ptr %src, i32 2)
  ret %struct.uint16x8x3_t %tmp
}

define %struct.uint32x4x3_t @test_vld3q_dup_u32(ptr %src) {
; CHECK-LABEL: test_vld3q_dup_u32:
; CHECK:         vld3.32 {d0[], d2[], d4[]}, [r0]
; CHECK-NEXT:    vld3.32 {d1[], d3[], d5[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint32x4x3_t @llvm.arm.neon.vld3dup.v4i32.p0(ptr %src, i32 4)
  ret %struct.uint32x4x3_t %tmp
}

define %struct.uint8x16x3_t @test_vld3q_dup_u8(ptr %src) {
; CHECK-LABEL: test_vld3q_dup_u8:
; CHECK:         vld3.8 {d0[], d2[], d4[]}, [r0]
; CHECK-NEXT:    vld3.8 {d1[], d3[], d5[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint8x16x3_t @llvm.arm.neon.vld3dup.v16i8.p0(ptr %src, i32 1)
  ret %struct.uint8x16x3_t %tmp
}

define %struct.uint16x8x4_t @test_vld4q_dup_u16(ptr %src) {
; CHECK-LABEL: test_vld4q_dup_u16:
; CHECK:         vld4.16 {d0[], d2[], d4[], d6[]}, [r0]
; CHECK-NEXT:    vld4.16 {d1[], d3[], d5[], d7[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint16x8x4_t @llvm.arm.neon.vld4dup.v8i16.p0(ptr %src, i32 2)
  ret %struct.uint16x8x4_t %tmp
}

define %struct.uint32x4x4_t @test_vld4q_dup_u32(ptr %src) {
; CHECK-LABEL: test_vld4q_dup_u32:
; CHECK:         vld4.32 {d0[], d2[], d4[], d6[]}, [r0]
; CHECK-NEXT:    vld4.32 {d1[], d3[], d5[], d7[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint32x4x4_t @llvm.arm.neon.vld4dup.v4i32.p0(ptr %src, i32 4)
  ret %struct.uint32x4x4_t %tmp
}

define %struct.uint8x16x4_t @test_vld4q_dup_u8(ptr %src) {
; CHECK-LABEL: test_vld4q_dup_u8:
; CHECK:         vld4.8 {d0[], d2[], d4[], d6[]}, [r0]
; CHECK-NEXT:    vld4.8 {d1[], d3[], d5[], d7[]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %tmp = tail call %struct.uint8x16x4_t @llvm.arm.neon.vld4dup.v16i8.p0(ptr %src, i32 1)
  ret %struct.uint8x16x4_t %tmp
}
