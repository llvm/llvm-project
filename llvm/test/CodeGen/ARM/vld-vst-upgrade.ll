; RUN: llc -mtriple=arm-eabi -mattr=+neon < %s | FileCheck %s

%struct.__neon_int32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.__neon_int32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_int32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }

; vld[1234] auto-upgrade tests

; CHECK-LABEL: test_vld1_upgrade:
; CHECK: vld1.32 {d16}, [r0]
define <2 x i32> @test_vld1_upgrade(ptr %ptr) {
  %tmp1 = call <2 x i32> @llvm.arm.neon.vld1.v2i32(ptr %ptr, i32 1)
  ret <2 x i32> %tmp1
}

declare <2 x i32> @llvm.arm.neon.vld1.v2i32(ptr, i32) nounwind readonly

; CHECK-LABEL: test_vld2_upgrade:
; CHECK: vld2.32 {d16, d17}, [r0]
define %struct.__neon_int32x2x2_t @test_vld2_upgrade(ptr %ptr) {
  %tmp1 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2.v2i32(ptr %ptr, i32 1)
  ret %struct.__neon_int32x2x2_t %tmp1
}

declare %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2.v2i32(ptr, i32) nounwind readonly

; CHECK-LABEL: test_vld3_upgrade:
; CHECK: vld3.32 {d16, d17, d18}, [r1]
define %struct.__neon_int32x2x3_t @test_vld3_upgrade(ptr %ptr) {
  %tmp1 = call %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3.v2i32(ptr %ptr, i32 1)
  ret %struct.__neon_int32x2x3_t %tmp1
}

declare %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3.v2i32(ptr, i32) nounwind readonly

; CHECK-LABEL: test_vld4_upgrade:
; CHECK: vld4.32 {d16, d17, d18, d19}, [r1]
define %struct.__neon_int32x2x4_t @test_vld4_upgrade(ptr %ptr) {
  %tmp1 = call %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4.v2i32(ptr %ptr, i32 1)
  ret %struct.__neon_int32x2x4_t %tmp1
}

declare %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4.v2i32(ptr, i32) nounwind readonly

; vld[234]lane auto-upgrade tests

; CHECK-LABEL: test_vld2lane_upgrade:
; CHECK: vld2.32 {d16[1], d17[1]}, [r0]
define %struct.__neon_int32x2x2_t @test_vld2lane_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B) {
  %tmp1 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, i32 1, i32 1)
  ret %struct.__neon_int32x2x2_t %tmp1
}

declare %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32(ptr, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly

; CHECK-LABEL: test_vld3lane_upgrade:
; CHECK: vld3.32 {d16[1], d17[1], d18[1]}, [r1]
define %struct.__neon_int32x2x3_t @test_vld3lane_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C) {
  %tmp1 = call %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3lane.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, i32 1, i32 1)
  ret %struct.__neon_int32x2x3_t %tmp1
}

declare %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3lane.v2i32(ptr, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly

; CHECK-LABEL: test_vld4lane_upgrade:
; CHECK: vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [r1]
define %struct.__neon_int32x2x4_t @test_vld4lane_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D) {
  %tmp1 = call %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, i32 1, i32 1)
  ret %struct.__neon_int32x2x4_t %tmp1
}

declare %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32(ptr, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly

; vst[1234] auto-upgrade tests

; CHECK-LABEL: test_vst1_upgrade:
; CHECK: vst1.32 {d16}, [r0]
define void @test_vst1_upgrade(ptr %ptr, <2 x i32> %A) {
  call void @llvm.arm.neon.vst1.v2i32(ptr %ptr, <2 x i32> %A, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst1.v2i32(ptr, <2 x i32>, i32) nounwind

; CHECK-LABEL: test_vst2_upgrade:
; CHECK: vst2.32 {d16, d17}, [r0]
define void @test_vst2_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B) {
  call void @llvm.arm.neon.vst2.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst2.v2i32(ptr, <2 x i32>, <2 x i32>, i32) nounwind

; CHECK-LABEL: test_vst3_upgrade:
; CHECK: vst3.32 {d16, d17, d18}, [r0]
define void @test_vst3_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C) {
  call void @llvm.arm.neon.vst3.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst3.v2i32(ptr, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind

; CHECK-LABEL: test_vst4_upgrade:
; CHECK: vst4.32 {d16, d17, d18, d19}, [r0]
define void @test_vst4_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D) {
  call void @llvm.arm.neon.vst4.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst4.v2i32(ptr, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind

; vst[234]lane auto-upgrade tests

; CHECK-LABEL: test_vst2lane_upgrade:
; CHECK: vst2.32 {d16[1], d17[1]}, [r0]
define void @test_vst2lane_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B) {
  call void @llvm.arm.neon.vst2lane.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, i32 1, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst2lane.v2i32(ptr, <2 x i32>, <2 x i32>, i32, i32) nounwind

; CHECK-LABEL: test_vst3lane_upgrade:
; CHECK: vst3.32 {d16[1], d17[1], d18[1]}, [r0]
define void @test_vst3lane_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C) {
  call void @llvm.arm.neon.vst3lane.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, i32 1, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst3lane.v2i32(ptr, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind

; CHECK-LABEL: test_vst4lane_upgrade:
; CHECK: vst4.32 {d16[1], d17[1], d18[1], d19[1]}, [r0]
define void @test_vst4lane_upgrade(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D) {
  call void @llvm.arm.neon.vst4lane.v2i32(ptr %ptr, <2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, i32 1, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst4lane.v2i32(ptr, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind
