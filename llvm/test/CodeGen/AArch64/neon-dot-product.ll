; RUN: llc -mtriple aarch64-none-linux-gnu -mattr=+dotprod    < %s | FileCheck %s
; RUN: llc -mtriple aarch64-none-linux-gnu -mcpu=cortex-a65   < %s | FileCheck %s
; RUN: llc -mtriple aarch64-none-linux-gnu -mcpu=cortex-a65ae < %s | FileCheck %s
; RUN: llc -mtriple aarch64-none-linux-gnu -mcpu=neoverse-e1  < %s | FileCheck %s
; RUN: llc -mtriple aarch64-none-linux-gnu -mcpu=neoverse-n1  < %s | FileCheck %s

declare <2 x i32> @llvm.aarch64.neon.udot.v2i32.v8i8(<2 x i32>, <8 x i8>, <8 x i8>)
declare <4 x i32> @llvm.aarch64.neon.udot.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>)
declare <2 x i32> @llvm.aarch64.neon.sdot.v2i32.v8i8(<2 x i32>, <8 x i8>, <8 x i8>)
declare <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>)

define <2 x i32> @test_vdot_u32(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) #0 {
entry:
; CHECK-LABEL: test_vdot_u32:
; CHECK: udot v0.2s, v1.8b, v2.8b
  %vdot1.i = call <2 x i32> @llvm.aarch64.neon.udot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_u32(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) #0 {
entry:
; CHECK-LABEL: test_vdotq_u32:
; CHECK: udot v0.4s, v1.16b, v2.16b
  %vdot1.i = call <4 x i32> @llvm.aarch64.neon.udot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) #2
  ret <4 x i32> %vdot1.i
}

define <2 x i32> @test_vdot_s32(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) #0 {
entry:
; CHECK-LABEL: test_vdot_s32:
; CHECK: sdot v0.2s, v1.8b, v2.8b
  %vdot1.i = call <2 x i32> @llvm.aarch64.neon.sdot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_s32(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) #0 {
entry:
; CHECK-LABEL: test_vdotq_s32:
; CHECK: sdot v0.4s, v1.16b, v2.16b
  %vdot1.i = call <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) #2
  ret <4 x i32> %vdot1.i
}

define <2 x i32> @test_vdot_lane_u32(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) {
entry:
; CHECK-LABEL: test_vdot_lane_u32:
; CHECK: udot v0.2s, v1.8b, v2.4b[1]
  %.cast = bitcast <8 x i8> %c to <2 x i32>
  %shuffle = shufflevector <2 x i32> %.cast, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %.cast5 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vdot1.i = call <2 x i32> @llvm.aarch64.neon.udot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %.cast5) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_lane_u32(<4 x i32> %a, <16 x i8> %b, <8 x i8> %c) {
entry:
; CHECK-LABEL: test_vdotq_lane_u32:
; CHECK:  udot v0.4s, v1.16b, v2.4b[1]
  %.cast = bitcast <8 x i8> %c to <2 x i32>
  %shuffle = shufflevector <2 x i32> %.cast, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %.cast3 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vdot1.i = call <4 x i32> @llvm.aarch64.neon.udot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %.cast3) #2
  ret <4 x i32> %vdot1.i
}

define <2 x i32> @test_vdot_laneq_u32(<2 x i32> %a, <8 x i8> %b, <16 x i8> %c) {
entry:
; CHECK-LABEL: test_vdot_laneq_u32:
; CHECK:  udot v0.2s, v1.8b, v2.4b[1]
  %.cast = bitcast <16 x i8> %c to <4 x i32>
  %shuffle = shufflevector <4 x i32> %.cast, <4 x i32> undef, <2 x i32> <i32 1, i32 1>
  %.cast5 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vdot1.i = call <2 x i32> @llvm.aarch64.neon.udot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %.cast5) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_laneq_u32(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) {
entry:
; CHECK-LABEL: test_vdotq_laneq_u32:
; CHECK:  udot v0.4s, v1.16b, v2.4b[1]
  %.cast = bitcast <16 x i8> %c to <4 x i32>
  %shuffle = shufflevector <4 x i32> %.cast, <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %.cast3 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vdot1.i = call <4 x i32> @llvm.aarch64.neon.udot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %.cast3) #2
  ret <4 x i32> %vdot1.i
}

define <2 x i32> @test_vdot_lane_s32(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) {
entry:
; CHECK-LABEL: test_vdot_lane_s32:
; CHECK: sdot v0.2s, v1.8b, v2.4b[1]
  %.cast = bitcast <8 x i8> %c to <2 x i32>
  %shuffle = shufflevector <2 x i32> %.cast, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %.cast5 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vdot1.i = call <2 x i32> @llvm.aarch64.neon.sdot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %.cast5) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_lane_s32(<4 x i32> %a, <16 x i8> %b, <8 x i8> %c) {
entry:
; CHECK-LABEL: test_vdotq_lane_s32:
; CHECK:  sdot v0.4s, v1.16b, v2.4b[1]
  %.cast = bitcast <8 x i8> %c to <2 x i32>
  %shuffle = shufflevector <2 x i32> %.cast, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %.cast3 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vdot1.i = call <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %.cast3) #2
  ret <4 x i32> %vdot1.i
}

define <2 x i32> @test_vdot_laneq_s32(<2 x i32> %a, <8 x i8> %b, <16 x i8> %c) {
entry:
; CHECK-LABEL: test_vdot_laneq_s32:
; CHECK:  sdot v0.2s, v1.8b, v2.4b[1]
  %.cast = bitcast <16 x i8> %c to <4 x i32>
  %shuffle = shufflevector <4 x i32> %.cast, <4 x i32> undef, <2 x i32> <i32 1, i32 1>
  %.cast5 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vdot1.i = call <2 x i32> @llvm.aarch64.neon.sdot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %.cast5) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_laneq_s32(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) {
entry:
; CHECK-LABEL: test_vdotq_laneq_s32:
; CHECK:  sdot v0.4s, v1.16b, v2.4b[1]
  %.cast = bitcast <16 x i8> %c to <4 x i32>
  %shuffle = shufflevector <4 x i32> %.cast, <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %.cast3 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vdot1.i = call <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %.cast3) #2
  ret <4 x i32> %vdot1.i
}

define fastcc void @test_sdot_v4i8(i8* noalias nocapture %0, i8* noalias nocapture readonly %1, i8* noalias nocapture readonly %2) {
entry:
; CHECK-LABEL: test_sdot_v4i8:
; CHECK:  sdot {{v[0-9]+}}.2s, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  %3 = bitcast i8* %0 to i32*
  %4 = load i8, i8* %1, align 1
  %5 = sext i8 %4 to i32
  %6 = load i8, i8* %2, align 1
  %7 = sext i8 %6 to i32
  %8 = mul nsw i32 %7, %5
  %9 = getelementptr inbounds i8, i8* %1, i64 1
  %10 = load i8, i8* %9, align 1
  %11 = sext i8 %10 to i32
  %12 = getelementptr inbounds i8, i8* %2, i64 1
  %13 = load i8, i8* %12, align 1
  %14 = sext i8 %13 to i32
  %15 = mul nsw i32 %14, %11
  %16 = add nsw i32 %15, %8
  %17 = getelementptr inbounds i8, i8* %1, i64 2
  %18 = load i8, i8* %17, align 1
  %19 = sext i8 %18 to i32
  %20 = getelementptr inbounds i8, i8* %2, i64 2
  %21 = load i8, i8* %20, align 1
  %22 = sext i8 %21 to i32
  %23 = mul nsw i32 %22, %19
  %24 = add nsw i32 %23, %16
  %25 = getelementptr inbounds i8, i8* %1, i64 3
  %26 = load i8, i8* %25, align 1
  %27 = sext i8 %26 to i32
  %28 = getelementptr inbounds i8, i8* %2, i64 3
  %29 = load i8, i8* %28, align 1
  %30 = sext i8 %29 to i32
  %31 = mul nsw i32 %30, %27
  %32 = add nsw i32 %31, %24
  store i32 %32, i32* %3, align 64
  ret void
}

define fastcc void @test_udot_v4i8(i8* noalias nocapture %0, i8* noalias nocapture readonly %1, i8* noalias nocapture readonly %2) {
entry:
; CHECK-LABEL: test_udot_v4i8:
; CHECK:  udot {{v[0-9]+}}.2s, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  %3 = bitcast i8* %0 to i32*
  %4 = load i8, i8* %1, align 1
  %5 = zext i8 %4 to i32
  %6 = load i8, i8* %2, align 1
  %7 = zext i8 %6 to i32
  %8 = mul nsw i32 %7, %5
  %9 = getelementptr inbounds i8, i8* %1, i64 1
  %10 = load i8, i8* %9, align 1
  %11 = zext i8 %10 to i32
  %12 = getelementptr inbounds i8, i8* %2, i64 1
  %13 = load i8, i8* %12, align 1
  %14 = zext i8 %13 to i32
  %15 = mul nsw i32 %14, %11
  %16 = add nsw i32 %15, %8
  %17 = getelementptr inbounds i8, i8* %1, i64 2
  %18 = load i8, i8* %17, align 1
  %19 = zext i8 %18 to i32
  %20 = getelementptr inbounds i8, i8* %2, i64 2
  %21 = load i8, i8* %20, align 1
  %22 = zext i8 %21 to i32
  %23 = mul nsw i32 %22, %19
  %24 = add nsw i32 %23, %16
  %25 = getelementptr inbounds i8, i8* %1, i64 3
  %26 = load i8, i8* %25, align 1
  %27 = zext i8 %26 to i32
  %28 = getelementptr inbounds i8, i8* %2, i64 3
  %29 = load i8, i8* %28, align 1
  %30 = zext i8 %29 to i32
  %31 = mul nsw i32 %30, %27
  %32 = add nsw i32 %31, %24
  store i32 %32, i32* %3, align 64
  ret void
}

declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)

define i32 @test_udot_v8i8(i8* nocapture readonly %a, i8* nocapture readonly %b) {
entry:
; CHECK-LABEL: test_udot_v8i8:
; CHECK:  udot {{v[0-9]+}}.2s, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  %0 = bitcast i8* %a to <8 x i8>*
  %1 = load <8 x i8>, <8 x i8>* %0
  %2 = zext <8 x i8> %1 to <8 x i32>
  %3 = bitcast i8* %b to <8 x i8>*
  %4 = load <8 x i8>, <8 x i8>* %3
  %5 = zext <8 x i8> %4 to <8 x i32>
  %6 = mul nuw nsw <8 x i32> %5, %2
  %7 = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %6)
  ret i32 %7
}

define i32 @test_sdot_v8i8(i8* nocapture readonly %a, i8* nocapture readonly %b) {
entry:
; CHECK-LABEL: test_sdot_v8i8:
; CHECK:  sdot {{v[0-9]+}}.2s, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  %0 = bitcast i8* %a to <8 x i8>*
  %1 = load <8 x i8>, <8 x i8>* %0
  %2 = sext <8 x i8> %1 to <8 x i32>
  %3 = bitcast i8* %b to <8 x i8>*
  %4 = load <8 x i8>, <8 x i8>* %3
  %5 = sext <8 x i8> %4 to <8 x i32>
  %6 = mul nsw <8 x i32> %5, %2
  %7 = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %6)
  ret i32 %7
}

declare i32 @llvm.vector.reduce.add.v16i32(<16 x i32>)

define i32 @test_udot_v16i8(i8* nocapture readonly %a, i8* nocapture readonly %b, i32 %sum) {
entry:
; CHECK-LABEL: test_udot_v16i8:
; CHECK:  udot {{v[0-9]+}}.4s, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  %0 = bitcast i8* %a to <16 x i8>*
  %1 = load <16 x i8>, <16 x i8>* %0
  %2 = zext <16 x i8> %1 to <16 x i32>
  %3 = bitcast i8* %b to <16 x i8>*
  %4 = load <16 x i8>, <16 x i8>* %3
  %5 = zext <16 x i8> %4 to <16 x i32>
  %6 = mul nuw nsw <16 x i32> %5, %2
  %7 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %6)
  %op.extra = add i32 %7, %sum
  ret i32 %op.extra
}

define i32 @test_udot_v16i8_2(i8* nocapture readonly %a1) {
; CHECK-LABEL: test_udot_v16i8_2:
; CHECK:    movi {{v[0-9]+}}.16b, #1
; CHECK:    movi {{v[0-9]+}}.2d, #0000000000000000
; CHECK:    udot {{v[0-9]+}}.4s, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK:    addv s0, {{v[0-9]+}}.4s
entry:
  %0 = bitcast i8* %a1 to <16 x i8>*
  %1 = load <16 x i8>, <16 x i8>* %0
  %2 = zext <16 x i8> %1 to <16 x i32>
  %3 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %2)
  ret i32 %3
}

define i32 @test_sdot_v16i8(i8* nocapture readonly %a, i8* nocapture readonly %b, i32 %sum) {
entry:
; CHECK-LABEL: test_sdot_v16i8:
; CHECK:  sdot {{v[0-9]+}}.4s, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  %0 = bitcast i8* %a to <16 x i8>*
  %1 = load <16 x i8>, <16 x i8>* %0
  %2 = sext <16 x i8> %1 to <16 x i32>
  %3 = bitcast i8* %b to <16 x i8>*
  %4 = load <16 x i8>, <16 x i8>* %3
  %5 = sext <16 x i8> %4 to <16 x i32>
  %6 = mul nsw <16 x i32> %5, %2
  %7 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %6)
  %op.extra = add nsw i32 %7, %sum
  ret i32 %op.extra
}

define i32 @test_sdot_v16i8_2(i8* nocapture readonly %a1) {
; CHECK-LABEL: test_sdot_v16i8_2:
; CHECK:    movi {{v[0-9]+}}.16b, #1
; CHECK:    movi {{v[0-9]+}}.2d, #0000000000000000
; CHECK:    sdot {{v[0-9]+}}.4s, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK:    addv s0, {{v[0-9]+}}.4s
entry:
  %0 = bitcast i8* %a1 to <16 x i8>*
  %1 = load <16 x i8>, <16 x i8>* %0
  %2 = sext <16 x i8> %1 to <16 x i32>
  %3 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %2)
  ret i32 %3
}
