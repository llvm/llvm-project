; RUN: llc < %s -mtriple=armv8-linux-gnueabi -verify-machineinstrs \
; RUN:     -asm-verbose=false | FileCheck %s

%struct.uint32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.uint32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.uint32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }

declare %struct.uint32x2x2_t @llvm.arm.neon.vld2dup.v2i32.p0i8(i8*, i32)
declare %struct.uint32x2x3_t @llvm.arm.neon.vld3dup.v2i32.p0i8(i8*, i32)
declare %struct.uint32x2x4_t @llvm.arm.neon.vld4dup.v2i32.p0i8(i8*, i32)

; CHECK-LABEL: test_vld2_dup_update
; CHECK: vld2.32 {d16[], d17[]}, {{\[}}[[SRC_R:r[0-9]+]]]
; CHECK: add {{r[0-9]+|lr}}, [[SRC_R]], #4
define i8* @test_vld2_dup_update(%struct.uint32x2x2_t* %dest, i8* %src) {
entry:
  %tmp = tail call %struct.uint32x2x2_t @llvm.arm.neon.vld2dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x2_t %tmp, %struct.uint32x2x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 4
  ret i8* %updated_src
}

; CHECK-LABEL: test_vld3_dup_update
; CHECK: vld3.32 {d16[], d17[], d18[]}, {{\[}}[[SRC_R:r[0-9]+]]]
; CHECK: add {{r[0-9]+|lr}}, [[SRC_R]], #4
define i8* @test_vld3_dup_update(%struct.uint32x2x3_t* %dest, i8* %src) {
entry:
  %tmp = tail call %struct.uint32x2x3_t @llvm.arm.neon.vld3dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x3_t %tmp, %struct.uint32x2x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 4
  ret i8* %updated_src
}

; CHECK-LABEL: test_vld4_dup_update
; CHECK: vld4.32 {d16[], d17[], d18[], d19[]}, {{\[}}[[SRC_R:r[0-9]+]]]
; CHECK: add {{r[0-9]+|lr}}, [[SRC_R]], #4
define i8* @test_vld4_dup_update(%struct.uint32x2x4_t* %dest, i8* %src) {
entry:
  %tmp = tail call %struct.uint32x2x4_t @llvm.arm.neon.vld4dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x4_t %tmp, %struct.uint32x2x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 4
  ret i8* %updated_src
}
