; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -global-isel -mtriple=arm64-eabi -aarch64-neon-syntax=apple -verify-machineinstrs | FileCheck %s
; The instruction latencies of Exynos-M3 trigger the transform we see under the Exynos check.
; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -verify-machineinstrs -mcpu=exynos-m3 | FileCheck --check-prefix=EXYNOS %s

define void @st1lane_16b(<16 x i8> %A, ptr %D) {
; CHECK-LABEL: st1lane_16b
; CHECK: st1.b { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr i8, ptr %D, i64 1
  %tmp = extractelement <16 x i8> %A, i32 1
  store i8 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_16b(<16 x i8> %A, ptr %D) {
; CHECK-LABEL: st1lane0_16b
; CHECK: st1.b { v0 }[0], [x{{[0-9]+}}]
  %ptr = getelementptr i8, ptr %D, i64 1
  %tmp = extractelement <16 x i8> %A, i32 0
  store i8 %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_16b(<16 x i8> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_16b
; CHECK: st1.b { v0 }[0], [x{{[0-9]+}}]
  %ptr = getelementptr i8, ptr %D, i64 -1
  %tmp = extractelement <16 x i8> %A, i32 0
  store i8 %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_16b(<16 x i8> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_16b
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.b { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr i8, ptr %D, i64 %offset
  %tmp = extractelement <16 x i8> %A, i32 1
  store i8 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_16b(<16 x i8> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_16b
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.b { v0 }[0], [x[[XREG]]]
  %ptr = getelementptr i8, ptr %D, i64 %offset
  %tmp = extractelement <16 x i8> %A, i32 0
  store i8 %tmp, ptr %ptr
  ret void
}

define void @st1lane_8h(<8 x i16> %A, ptr %D) {
; CHECK-LABEL: st1lane_8h
; CHECK: st1.h { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr i16, ptr %D, i64 1
  %tmp = extractelement <8 x i16> %A, i32 1
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_8h(<8 x i16> %A, ptr %D) {
; CHECK-LABEL: st1lane0_8h
; CHECK: str h0, [x0, #2]
  %ptr = getelementptr i16, ptr %D, i64 1
  %tmp = extractelement <8 x i16> %A, i32 0
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_8h(<8 x i16> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_8h
; CHECK: stur h0, [x0, #-2]
  %ptr = getelementptr i16, ptr %D, i64 -1
  %tmp = extractelement <8 x i16> %A, i32 0
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_8h(<8 x i16> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_8h
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.h { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr i16, ptr %D, i64 %offset
  %tmp = extractelement <8 x i16> %A, i32 1
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_8h(<8 x i16> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_8h
; CHECK: str h0, [x0, x1, lsl #1]
  %ptr = getelementptr i16, ptr %D, i64 %offset
  %tmp = extractelement <8 x i16> %A, i32 0
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane_4s(<4 x i32> %A, ptr %D) {
; CHECK-LABEL: st1lane_4s
; CHECK: st1.s { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr i32, ptr %D, i64 1
  %tmp = extractelement <4 x i32> %A, i32 1
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_4s(<4 x i32> %A, ptr %D) {
; CHECK-LABEL: st1lane0_4s
; CHECK: str s0, [x0, #4]
  %ptr = getelementptr i32, ptr %D, i64 1
  %tmp = extractelement <4 x i32> %A, i32 0
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_4s(<4 x i32> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_4s
; CHECK: stur s0, [x0, #-4]
  %ptr = getelementptr i32, ptr %D, i64 -1
  %tmp = extractelement <4 x i32> %A, i32 0
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_4s(<4 x i32> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_4s
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.s { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr i32, ptr %D, i64 %offset
  %tmp = extractelement <4 x i32> %A, i32 1
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_4s(<4 x i32> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_4s
; CHECK: str s0, [x0, x1, lsl #2]
  %ptr = getelementptr i32, ptr %D, i64 %offset
  %tmp = extractelement <4 x i32> %A, i32 0
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane_4s_float(<4 x float> %A, ptr %D) {
; CHECK-LABEL: st1lane_4s_float
; CHECK: st1.s { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr float, ptr %D, i64 1
  %tmp = extractelement <4 x float> %A, i32 1
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane0_4s_float(<4 x float> %A, ptr %D) {
; CHECK-LABEL: st1lane0_4s_float
; CHECK: str s0, [x0, #4]
  %ptr = getelementptr float, ptr %D, i64 1
  %tmp = extractelement <4 x float> %A, i32 0
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_4s_float(<4 x float> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_4s_float
; CHECK: stur s0, [x0, #-4]
  %ptr = getelementptr float, ptr %D, i64 -1
  %tmp = extractelement <4 x float> %A, i32 0
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_4s_float(<4 x float> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_4s_float
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.s { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr float, ptr %D, i64 %offset
  %tmp = extractelement <4 x float> %A, i32 1
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_4s_float(<4 x float> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_4s_float
; CHECK: str s0, [x0, x1, lsl #2]
  %ptr = getelementptr float, ptr %D, i64 %offset
  %tmp = extractelement <4 x float> %A, i32 0
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane_2d(<2 x i64> %A, ptr %D) {
; CHECK-LABEL: st1lane_2d
; CHECK: st1.d { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr i64, ptr %D, i64 1
  %tmp = extractelement <2 x i64> %A, i32 1
  store i64 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_2d(<2 x i64> %A, ptr %D) {
; CHECK-LABEL: st1lane0_2d
; CHECK: str d0, [x0, #8]
  %ptr = getelementptr i64, ptr %D, i64 1
  %tmp = extractelement <2 x i64> %A, i32 0
  store i64 %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_2d(<2 x i64> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_2d
; CHECK: stur d0, [x0, #-8]
  %ptr = getelementptr i64, ptr %D, i64 -1
  %tmp = extractelement <2 x i64> %A, i32 0
  store i64 %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_2d(<2 x i64> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_2d
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.d { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr i64, ptr %D, i64 %offset
  %tmp = extractelement <2 x i64> %A, i32 1
  store i64 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_2d(<2 x i64> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_2d
; CHECK: str d0, [x0, x1, lsl #3]
  %ptr = getelementptr i64, ptr %D, i64 %offset
  %tmp = extractelement <2 x i64> %A, i32 0
  store i64 %tmp, ptr %ptr
  ret void
}

define void @st1lane_2d_double(<2 x double> %A, ptr %D) {
; CHECK-LABEL: st1lane_2d_double
; CHECK: st1.d { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr double, ptr %D, i64 1
  %tmp = extractelement <2 x double> %A, i32 1
  store double %tmp, ptr %ptr
  ret void
}

define void @st1lane0_2d_double(<2 x double> %A, ptr %D) {
; CHECK-LABEL: st1lane0_2d_double
; CHECK: str d0, [x0, #8]
  %ptr = getelementptr double, ptr %D, i64 1
  %tmp = extractelement <2 x double> %A, i32 0
  store double %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_2d_double(<2 x double> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_2d_double
; CHECK: stur d0, [x0, #-8]
  %ptr = getelementptr double, ptr %D, i64 -1
  %tmp = extractelement <2 x double> %A, i32 0
  store double %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_2d_double(<2 x double> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_2d_double
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.d { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr double, ptr %D, i64 %offset
  %tmp = extractelement <2 x double> %A, i32 1
  store double %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_2d_double(<2 x double> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_2d_double
; CHECK: str d0, [x0, x1, lsl #3]
  %ptr = getelementptr double, ptr %D, i64 %offset
  %tmp = extractelement <2 x double> %A, i32 0
  store double %tmp, ptr %ptr
  ret void
}

define void @st1lane_8b(<8 x i8> %A, ptr %D) {
; CHECK-LABEL: st1lane_8b
; CHECK: st1.b { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr i8, ptr %D, i64 1
  %tmp = extractelement <8 x i8> %A, i32 1
  store i8 %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_8b(<8 x i8> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_8b
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.b { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr i8, ptr %D, i64 %offset
  %tmp = extractelement <8 x i8> %A, i32 1
  store i8 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_8b(<8 x i8> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_8b
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.b { v0 }[0], [x[[XREG]]]
  %ptr = getelementptr i8, ptr %D, i64 %offset
  %tmp = extractelement <8 x i8> %A, i32 0
  store i8 %tmp, ptr %ptr
  ret void
}

define void @st1lane_4h(<4 x i16> %A, ptr %D) {
; CHECK-LABEL: st1lane_4h
; CHECK: st1.h { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr i16, ptr %D, i64 1
  %tmp = extractelement <4 x i16> %A, i32 1
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_4h(<4 x i16> %A, ptr %D) {
; CHECK-LABEL: st1lane0_4h
; CHECK: str h0, [x0, #2]
  %ptr = getelementptr i16, ptr %D, i64 1
  %tmp = extractelement <4 x i16> %A, i32 0
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_4h(<4 x i16> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_4h
; CHECK: stur h0, [x0, #-2]
  %ptr = getelementptr i16, ptr %D, i64 -1
  %tmp = extractelement <4 x i16> %A, i32 0
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_4h(<4 x i16> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_4h
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.h { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr i16, ptr %D, i64 %offset
  %tmp = extractelement <4 x i16> %A, i32 1
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_4h(<4 x i16> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_4h
; CHECK: str h0, [x0, x1, lsl #1]
  %ptr = getelementptr i16, ptr %D, i64 %offset
  %tmp = extractelement <4 x i16> %A, i32 0
  store i16 %tmp, ptr %ptr
  ret void
}

define void @st1lane_2s(<2 x i32> %A, ptr %D) {
; CHECK-LABEL: st1lane_2s
; CHECK: st1.s { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr i32, ptr %D, i64 1
  %tmp = extractelement <2 x i32> %A, i32 1
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_2s(<2 x i32> %A, ptr %D) {
; CHECK-LABEL: st1lane0_2s
; CHECK: str s0, [x0, #4]
  %ptr = getelementptr i32, ptr %D, i64 1
  %tmp = extractelement <2 x i32> %A, i32 0
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_2s(<2 x i32> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_2s
; CHECK: stur s0, [x0, #-4]
  %ptr = getelementptr i32, ptr %D, i64 -1
  %tmp = extractelement <2 x i32> %A, i32 0
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_2s(<2 x i32> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_2s
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.s { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr i32, ptr %D, i64 %offset
  %tmp = extractelement <2 x i32> %A, i32 1
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_2s(<2 x i32> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_2s
; CHECK: str s0, [x0, x1, lsl #2]
  %ptr = getelementptr i32, ptr %D, i64 %offset
  %tmp = extractelement <2 x i32> %A, i32 0
  store i32 %tmp, ptr %ptr
  ret void
}

define void @st1lane_2s_float(<2 x float> %A, ptr %D) {
; CHECK-LABEL: st1lane_2s_float
; CHECK: st1.s { v0 }[1], [x{{[0-9]+}}]
  %ptr = getelementptr float, ptr %D, i64 1
  %tmp = extractelement <2 x float> %A, i32 1
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane0_2s_float(<2 x float> %A, ptr %D) {
; CHECK-LABEL: st1lane0_2s_float
; CHECK: str s0, [x0, #4]
  %ptr = getelementptr float, ptr %D, i64 1
  %tmp = extractelement <2 x float> %A, i32 0
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_2s_float(<2 x float> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_2s_float
; CHECK: stur s0, [x0, #-4]
  %ptr = getelementptr float, ptr %D, i64 -1
  %tmp = extractelement <2 x float> %A, i32 0
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane_ro_2s_float(<2 x float> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane_ro_2s_float
; CHECK: add x[[XREG:[0-9]+]], x0, x1
; CHECK: st1.s { v0 }[1], [x[[XREG]]]
  %ptr = getelementptr float, ptr %D, i64 %offset
  %tmp = extractelement <2 x float> %A, i32 1
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_2s_float(<2 x float> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_2s_float
; CHECK: str s0, [x0, x1, lsl #2]
  %ptr = getelementptr float, ptr %D, i64 %offset
  %tmp = extractelement <2 x float> %A, i32 0
  store float %tmp, ptr %ptr
  ret void
}

define void @st1lane0_1d(<1 x i64> %A, ptr %D) {
; CHECK-LABEL: st1lane0_1d
; CHECK: str d0, [x0, #8]
  %ptr = getelementptr i64, ptr %D, i64 1
  %tmp = extractelement <1 x i64> %A, i32 0
  store i64 %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_1d(<1 x i64> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_1d
; CHECK: stur d0, [x0, #-8]
  %ptr = getelementptr i64, ptr %D, i64 -1
  %tmp = extractelement <1 x i64> %A, i32 0
  store i64 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_1d(<1 x i64> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_1d
; CHECK: str d0, [x0, x1, lsl #3]
  %ptr = getelementptr i64, ptr %D, i64 %offset
  %tmp = extractelement <1 x i64> %A, i32 0
  store i64 %tmp, ptr %ptr
  ret void
}

define void @st1lane0_1d_double(<1 x double> %A, ptr %D) {
; CHECK-LABEL: st1lane0_1d_double
; CHECK: str d0, [x0, #8]
  %ptr = getelementptr double, ptr %D, i64 1
  %tmp = extractelement <1 x double> %A, i32 0
  store double %tmp, ptr %ptr
  ret void
}

define void @st1lane0u_1d_double(<1 x double> %A, ptr %D) {
; CHECK-LABEL: st1lane0u_1d_double
; CHECK: stur d0, [x0, #-8]
  %ptr = getelementptr double, ptr %D, i64 -1
  %tmp = extractelement <1 x double> %A, i32 0
  store double %tmp, ptr %ptr
  ret void
}

define void @st1lane0_ro_1d_double(<1 x double> %A, ptr %D, i64 %offset) {
; CHECK-LABEL: st1lane0_ro_1d_double
; CHECK: str d0, [x0, x1, lsl #3]
  %ptr = getelementptr double, ptr %D, i64 %offset
  %tmp = extractelement <1 x double> %A, i32 0
  store double %tmp, ptr %ptr
  ret void
}

define void @st2lane_16b(<16 x i8> %A, <16 x i8> %B, ptr %D) {
; CHECK-LABEL: st2lane_16b
; CHECK: st2.b
  call void @llvm.aarch64.neon.st2lane.v16i8.p0(<16 x i8> %A, <16 x i8> %B, i64 1, ptr %D)
  ret void
}

define void @st2lane_8h(<8 x i16> %A, <8 x i16> %B, ptr %D) {
; CHECK-LABEL: st2lane_8h
; CHECK: st2.h
  call void @llvm.aarch64.neon.st2lane.v8i16.p0(<8 x i16> %A, <8 x i16> %B, i64 1, ptr %D)
  ret void
}

define void @st2lane_4s(<4 x i32> %A, <4 x i32> %B, ptr %D) {
; CHECK-LABEL: st2lane_4s
; CHECK: st2.s
  call void @llvm.aarch64.neon.st2lane.v4i32.p0(<4 x i32> %A, <4 x i32> %B, i64 1, ptr %D)
  ret void
}

define void @st2lane_2d(<2 x i64> %A, <2 x i64> %B, ptr %D) {
; CHECK-LABEL: st2lane_2d
; CHECK: st2.d
  call void @llvm.aarch64.neon.st2lane.v2i64.p0(<2 x i64> %A, <2 x i64> %B, i64 1, ptr %D)
  ret void
}

declare void @llvm.aarch64.neon.st2lane.v16i8.p0(<16 x i8>, <16 x i8>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st2lane.v8i16.p0(<8 x i16>, <8 x i16>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st2lane.v4i32.p0(<4 x i32>, <4 x i32>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st2lane.v2i64.p0(<2 x i64>, <2 x i64>, i64, ptr) nounwind readnone

define void @st3lane_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, ptr %D) {
; CHECK-LABEL: st3lane_16b
; CHECK: st3.b
  call void @llvm.aarch64.neon.st3lane.v16i8.p0(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, i64 1, ptr %D)
  ret void
}

define void @st3lane_8h(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, ptr %D) {
; CHECK-LABEL: st3lane_8h
; CHECK: st3.h
  call void @llvm.aarch64.neon.st3lane.v8i16.p0(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, i64 1, ptr %D)
  ret void
}

define void @st3lane_4s(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, ptr %D) {
; CHECK-LABEL: st3lane_4s
; CHECK: st3.s
  call void @llvm.aarch64.neon.st3lane.v4i32.p0(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, i64 1, ptr %D)
  ret void
}

define void @st3lane_2d(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %D) {
; CHECK-LABEL: st3lane_2d
; CHECK: st3.d
  call void @llvm.aarch64.neon.st3lane.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, i64 1, ptr %D)
  ret void
}

declare void @llvm.aarch64.neon.st3lane.v16i8.p0(<16 x i8>, <16 x i8>, <16 x i8>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st3lane.v8i16.p0(<8 x i16>, <8 x i16>, <8 x i16>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st3lane.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st3lane.v2i64.p0(<2 x i64>, <2 x i64>, <2 x i64>, i64, ptr) nounwind readnone

define void @st4lane_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, ptr %E) {
; CHECK-LABEL: st4lane_16b
; CHECK: st4.b
  call void @llvm.aarch64.neon.st4lane.v16i8.p0(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, i64 1, ptr %E)
  ret void
}

define void @st4lane_8h(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, <8 x i16> %D, ptr %E) {
; CHECK-LABEL: st4lane_8h
; CHECK: st4.h
  call void @llvm.aarch64.neon.st4lane.v8i16.p0(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, <8 x i16> %D, i64 1, ptr %E)
  ret void
}

define void @st4lane_4s(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, <4 x i32> %D, ptr %E) {
; CHECK-LABEL: st4lane_4s
; CHECK: st4.s
  call void @llvm.aarch64.neon.st4lane.v4i32.p0(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, <4 x i32> %D, i64 1, ptr %E)
  ret void
}

define void @st4lane_2d(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %E) {
; CHECK-LABEL: st4lane_2d
; CHECK: st4.d
  call void @llvm.aarch64.neon.st4lane.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, i64 1, ptr %E)
  ret void
}

declare void @llvm.aarch64.neon.st4lane.v16i8.p0(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st4lane.v8i16.p0(<8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st4lane.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i64, ptr) nounwind readnone
declare void @llvm.aarch64.neon.st4lane.v2i64.p0(<2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, i64, ptr) nounwind readnone


define void @st2_8b(<8 x i8> %A, <8 x i8> %B, ptr %P) nounwind {
; CHECK-LABEL: st2_8b
; CHECK: st2.8b
; EXYNOS-LABEL: st2_8b
; EXYNOS: zip1.8b
; EXYNOS: zip2.8b
; EXYNOS: stp
	call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %A, <8 x i8> %B, ptr %P)
	ret void
}

define void @st3_8b(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, ptr %P) nounwind {
; CHECK-LABEL: st3_8b
; CHECK: st3.8b
	call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, ptr %P)
	ret void
}

define void @st4_8b(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, <8 x i8> %D, ptr %P) nounwind {
; CHECK-LABEL: st4_8b
; CHECK: st4.8b
; EXYNOS-LABEL: st4_8b
; EXYNOS: zip1.8b
; EXYNOS: zip2.8b
; EXYNOS: zip1.8b
; EXYNOS: zip2.8b
; EXYNOS: zip1.8b
; EXYNOS: zip2.8b
; EXYNOS: stp
; EXYNOS: zip1.8b
; EXYNOS: zip2.8b
; EXYNOS: stp
	call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, <8 x i8> %D, ptr %P)
	ret void
}

declare void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8>, <8 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8>, <8 x i8>, <8 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, ptr) nounwind readonly

define void @st2_16b(<16 x i8> %A, <16 x i8> %B, ptr %P) nounwind {
; CHECK-LABEL: st2_16b
; CHECK: st2.16b
; EXYNOS-LABEL: st2_16b
; EXYNOS: zip1.16b
; EXYNOS: zip2.16b
; EXYNOS: stp
	call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> %A, <16 x i8> %B, ptr %P)
	ret void
}

define void @st3_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, ptr %P) nounwind {
; CHECK-LABEL: st3_16b
; CHECK: st3.16b
	call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, ptr %P)
	ret void
}

define void @st4_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, ptr %P) nounwind {
; CHECK-LABEL: st4_16b
; CHECK: st4.16b
; EXYNOS-LABEL: st4_16b
; EXYNOS: zip1.16b
; EXYNOS: zip2.16b
; EXYNOS: zip1.16b
; EXYNOS: zip2.16b
; EXYNOS: zip1.16b
; EXYNOS: zip2.16b
; EXYNOS: stp
; EXYNOS: zip1.16b
; EXYNOS: zip2.16b
; EXYNOS: stp
	call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, ptr %P)
	ret void
}

declare void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8>, <16 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8>, <16 x i8>, <16 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, ptr) nounwind readonly

define void @st2_4h(<4 x i16> %A, <4 x i16> %B, ptr %P) nounwind {
; CHECK-LABEL: st2_4h
; CHECK: st2.4h
; EXYNOS-LABEL: st2_4h
; EXYNOS: zip1.4h
; EXYNOS: zip2.4h
; EXYNOS: stp
	call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> %A, <4 x i16> %B, ptr %P)
	ret void
}

define void @st3_4h(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, ptr %P) nounwind {
; CHECK-LABEL: st3_4h
; CHECK: st3.4h
	call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, ptr %P)
	ret void
}

define void @st4_4h(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, <4 x i16> %D, ptr %P) nounwind {
; CHECK-LABEL: st4_4h
; CHECK: st4.4h
; EXYNOS-LABEL: st4_4h
; EXYNOS: zip1.4h
; EXYNOS: zip2.4h
; EXYNOS: zip1.4h
; EXYNOS: zip2.4h
; EXYNOS: zip1.4h
; EXYNOS: zip2.4h
; EXYNOS: stp
; EXYNOS: zip1.4h
; EXYNOS: zip2.4h
; EXYNOS: stp
	call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, <4 x i16> %D, ptr %P)
	ret void
}

declare void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16>, <4 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16>, <4 x i16>, <4 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, ptr) nounwind readonly

define void @st2_8h(<8 x i16> %A, <8 x i16> %B, ptr %P) nounwind {
; CHECK-LABEL: st2_8h
; CHECK: st2.8h
; EXYNOS-LABEL: st2_8h
; EXYNOS: zip1.8h
; EXYNOS: zip2.8h
; EXYNOS: stp
	call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> %A, <8 x i16> %B, ptr %P)
	ret void
}

define void @st3_8h(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, ptr %P) nounwind {
; CHECK-LABEL: st3_8h
; CHECK: st3.8h
	call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, ptr %P)
	ret void
}

define void @st4_8h(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, <8 x i16> %D, ptr %P) nounwind {
; CHECK-LABEL: st4_8h
; CHECK: st4.8h
; EXYNOS-LABEL: st4_8h
; EXYNOS: zip1.8h
; EXYNOS: zip2.8h
; EXYNOS: zip1.8h
; EXYNOS: zip2.8h
; EXYNOS: zip1.8h
; EXYNOS: zip2.8h
; EXYNOS: stp
; EXYNOS: zip1.8h
; EXYNOS: zip2.8h
; EXYNOS: stp
	call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, <8 x i16> %D, ptr %P)
	ret void
}

declare void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16>, <8 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16>, <8 x i16>, <8 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, ptr) nounwind readonly

define void @st2_2s(<2 x i32> %A, <2 x i32> %B, ptr %P) nounwind {
; CHECK-LABEL: st2_2s
; CHECK: st2.2s
; EXYNOS-LABEL: st2_2s
; EXYNOS: zip1.2s
; EXYNOS: zip2.2s
; EXYNOS: stp
	call void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32> %A, <2 x i32> %B, ptr %P)
	ret void
}

define void @st3_2s(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, ptr %P) nounwind {
; CHECK-LABEL: st3_2s
; CHECK: st3.2s
	call void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, ptr %P)
	ret void
}

define void @st4_2s(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, ptr %P) nounwind {
; CHECK-LABEL: st4_2s
; CHECK: st4.2s
; EXYNOS-LABEL: st4_2s
; EXYNOS: zip1.2s
; EXYNOS: zip2.2s
; EXYNOS: zip1.2s
; EXYNOS: zip2.2s
; EXYNOS: zip1.2s
; EXYNOS: zip2.2s
; EXYNOS: stp
; EXYNOS: zip1.2s
; EXYNOS: zip2.2s
; EXYNOS: stp
	call void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, ptr %P)
	ret void
}

declare void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32>, <2 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32>, <2 x i32>, <2 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, ptr) nounwind readonly

define void @st2_4s(<4 x i32> %A, <4 x i32> %B, ptr %P) nounwind {
; CHECK-LABEL: st2_4s
; CHECK: st2.4s
; EXYNOS-LABEL: st2_4s
; EXYNOS: zip1.4s
; EXYNOS: zip2.4s
; EXYNOS: stp
	call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %A, <4 x i32> %B, ptr %P)
	ret void
}

define void @st3_4s(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, ptr %P) nounwind {
; CHECK-LABEL: st3_4s
; CHECK: st3.4s
	call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, ptr %P)
	ret void
}

define void @st4_4s(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, <4 x i32> %D, ptr %P) nounwind {
; CHECK-LABEL: st4_4s
; CHECK: st4.4s
; EXYNOS-LABEL: st4_4s
; EXYNOS: zip1.4s
; EXYNOS: zip2.4s
; EXYNOS: zip1.4s
; EXYNOS: zip2.4s
; EXYNOS: zip1.4s
; EXYNOS: zip2.4s
; EXYNOS: stp
; EXYNOS: zip1.4s
; EXYNOS: zip2.4s
; EXYNOS: stp
	call void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, <4 x i32> %D, ptr %P)
	ret void
}

declare void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32>, <4 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, ptr) nounwind readonly

; If there's only one element, st2/3/4 don't make much sense, stick to st1.
define void @st2_1d(<1 x i64> %A, <1 x i64> %B, ptr %P) nounwind {
; CHECK-LABEL: st2_1d
; CHECK: st1.1d
	call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> %A, <1 x i64> %B, ptr %P)
	ret void
}

define void @st3_1d(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, ptr %P) nounwind {
; CHECK-LABEL: st3_1d
; CHECK: st1.1d
	call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, ptr %P)
	ret void
}

define void @st4_1d(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, <1 x i64> %D, ptr %P) nounwind {
; CHECK-LABEL: st4_1d
; CHECK: st1.1d
	call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, <1 x i64> %D, ptr %P)
	ret void
}

declare void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64>, <1 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64>, <1 x i64>, <1 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, ptr) nounwind readonly

define void @st2_2d(<2 x i64> %A, <2 x i64> %B, ptr %P) nounwind {
; CHECK-LABEL: st2_2d
; CHECK: st2.2d
; EXYNOS-LABEL: st2_2d
; EXYNOS: zip1.2d
; EXYNOS: zip2.2d
; EXYNOS: stp
	call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> %A, <2 x i64> %B, ptr %P)
	ret void
}

define void @st3_2d(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %P) nounwind {
; CHECK-LABEL: st3_2d
; CHECK: st3.2d
	call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %P)
	ret void
}

define void @st4_2d(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %P) nounwind {
; CHECK-LABEL: st4_2d
; CHECK: st4.2d
; EXYNOS-LABEL: st4_2d
; EXYNOS: zip1.2d
; EXYNOS: zip2.2d
; EXYNOS: zip1.2d
; EXYNOS: zip2.2d
; EXYNOS: zip1.2d
; EXYNOS: zip2.2d
; EXYNOS: stp
; EXYNOS: zip1.2d
; EXYNOS: zip2.2d
; EXYNOS: stp
	call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %P)
	ret void
}

declare void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64>, <2 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64>, <2 x i64>, <2 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, ptr) nounwind readonly

declare void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8>, <8 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v4i16.p0(<4 x i16>, <4 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v2i32.p0(<2 x i32>, <2 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v2f32.p0(<2 x float>, <2 x float>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64>, <1 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v1f64.p0(<1 x double>, <1 x double>, ptr) nounwind readonly

define void @st1_x2_v8i8(<8 x i8> %A, <8 x i8> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v8i8:
; CHECK: st1.8b { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8> %A, <8 x i8> %B, ptr %addr)
  ret void
}

define void @st1_x2_v4i16(<4 x i16> %A, <4 x i16> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v4i16:
; CHECK: st1.4h { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v4i16.p0(<4 x i16> %A, <4 x i16> %B, ptr %addr)
  ret void
}

define void @st1_x2_v2i32(<2 x i32> %A, <2 x i32> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v2i32:
; CHECK: st1.2s { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v2i32.p0(<2 x i32> %A, <2 x i32> %B, ptr %addr)
  ret void
}

define void @st1_x2_v2f32(<2 x float> %A, <2 x float> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v2f32:
; CHECK: st1.2s { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v2f32.p0(<2 x float> %A, <2 x float> %B, ptr %addr)
  ret void
}

define void @st1_x2_v1i64(<1 x i64> %A, <1 x i64> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v1i64:
; CHECK: st1.1d { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> %A, <1 x i64> %B, ptr %addr)
  ret void
}

define void @st1_x2_v1f64(<1 x double> %A, <1 x double> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v1f64:
; CHECK: st1.1d { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v1f64.p0(<1 x double> %A, <1 x double> %B, ptr %addr)
  ret void
}

declare void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8>, <16 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v8i16.p0(<8 x i16>, <8 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v4i32.p0(<4 x i32>, <4 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v4f32.p0(<4 x float>, <4 x float>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64>, <2 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x2.v2f64.p0(<2 x double>, <2 x double>, ptr) nounwind readonly

define void @st1_x2_v16i8(<16 x i8> %A, <16 x i8> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v16i8:
; CHECK: st1.16b { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8> %A, <16 x i8> %B, ptr %addr)
  ret void
}

define void @st1_x2_v8i16(<8 x i16> %A, <8 x i16> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v8i16:
; CHECK: st1.8h { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v8i16.p0(<8 x i16> %A, <8 x i16> %B, ptr %addr)
  ret void
}

define void @st1_x2_v4i32(<4 x i32> %A, <4 x i32> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v4i32:
; CHECK: st1.4s { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v4i32.p0(<4 x i32> %A, <4 x i32> %B, ptr %addr)
  ret void
}

define void @st1_x2_v4f32(<4 x float> %A, <4 x float> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v4f32:
; CHECK: st1.4s { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v4f32.p0(<4 x float> %A, <4 x float> %B, ptr %addr)
  ret void
}

define void @st1_x2_v2i64(<2 x i64> %A, <2 x i64> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v2i64:
; CHECK: st1.2d { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> %A, <2 x i64> %B, ptr %addr)
  ret void
}

define void @st1_x2_v2f64(<2 x double> %A, <2 x double> %B, ptr %addr) {
; CHECK-LABEL: st1_x2_v2f64:
; CHECK: st1.2d { {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x2.v2f64.p0(<2 x double> %A, <2 x double> %B, ptr %addr)
  ret void
}

declare void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8>, <8 x i8>, <8 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v4i16.p0(<4 x i16>, <4 x i16>, <4 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v2i32.p0(<2 x i32>, <2 x i32>, <2 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v2f32.p0(<2 x float>, <2 x float>, <2 x float>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64>, <1 x i64>, <1 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v1f64.p0(<1 x double>, <1 x double>, <1 x double>, ptr) nounwind readonly

define void @st1_x3_v8i8(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v8i8:
; CHECK: st1.8b { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, ptr %addr)
  ret void
}

define void @st1_x3_v4i16(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v4i16:
; CHECK: st1.4h { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v4i16.p0(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, ptr %addr)
  ret void
}

define void @st1_x3_v2i32(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v2i32:
; CHECK: st1.2s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v2i32.p0(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, ptr %addr)
  ret void
}

define void @st1_x3_v2f32(<2 x float> %A, <2 x float> %B, <2 x float> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v2f32:
; CHECK: st1.2s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v2f32.p0(<2 x float> %A, <2 x float> %B, <2 x float> %C, ptr %addr)
  ret void
}

define void @st1_x3_v1i64(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v1i64:
; CHECK: st1.1d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, ptr %addr)
  ret void
}

define void @st1_x3_v1f64(<1 x double> %A, <1 x double> %B, <1 x double> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v1f64:
; CHECK: st1.1d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v1f64.p0(<1 x double> %A, <1 x double> %B, <1 x double> %C, ptr %addr)
  ret void
}

declare void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8>, <16 x i8>, <16 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v8i16.p0(<8 x i16>, <8 x i16>, <8 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v4f32.p0(<4 x float>, <4 x float>, <4 x float>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64>, <2 x i64>, <2 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x3.v2f64.p0(<2 x double>, <2 x double>, <2 x double>, ptr) nounwind readonly

define void @st1_x3_v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v16i8:
; CHECK: st1.16b { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, ptr %addr)
  ret void
}

define void @st1_x3_v8i16(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v8i16:
; CHECK: st1.8h { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v8i16.p0(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, ptr %addr)
  ret void
}

define void @st1_x3_v4i32(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v4i32:
; CHECK: st1.4s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v4i32.p0(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, ptr %addr)
  ret void
}

define void @st1_x3_v4f32(<4 x float> %A, <4 x float> %B, <4 x float> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v4f32:
; CHECK: st1.4s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v4f32.p0(<4 x float> %A, <4 x float> %B, <4 x float> %C, ptr %addr)
  ret void
}

define void @st1_x3_v2i64(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v2i64:
; CHECK: st1.2d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %addr)
  ret void
}

define void @st1_x3_v2f64(<2 x double> %A, <2 x double> %B, <2 x double> %C, ptr %addr) {
; CHECK-LABEL: st1_x3_v2f64:
; CHECK: st1.2d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x3.v2f64.p0(<2 x double> %A, <2 x double> %B, <2 x double> %C, ptr %addr)
  ret void
}


declare void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v4i16.p0(<4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v2i32.p0(<2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v2f32.p0(<2 x float>, <2 x float>, <2 x float>, <2 x float>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v1f64.p0(<1 x double>, <1 x double>, <1 x double>, <1 x double>, ptr) nounwind readonly

define void @st1_x4_v8i8(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, <8 x i8> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v8i8:
; CHECK: st1.8b { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, <8 x i8> %D, ptr %addr)
  ret void
}

define void @st1_x4_v4i16(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, <4 x i16> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v4i16:
; CHECK: st1.4h { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v4i16.p0(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, <4 x i16> %D, ptr %addr)
  ret void
}

define void @st1_x4_v2i32(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v2i32:
; CHECK: st1.2s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v2i32.p0(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, ptr %addr)
  ret void
}

define void @st1_x4_v2f32(<2 x float> %A, <2 x float> %B, <2 x float> %C, <2 x float> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v2f32:
; CHECK: st1.2s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v2f32.p0(<2 x float> %A, <2 x float> %B, <2 x float> %C, <2 x float> %D, ptr %addr)
  ret void
}

define void @st1_x4_v1i64(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, <1 x i64> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v1i64:
; CHECK: st1.1d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, <1 x i64> %D, ptr %addr)
  ret void
}

define void @st1_x4_v1f64(<1 x double> %A, <1 x double> %B, <1 x double> %C, <1 x double> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v1f64:
; CHECK: st1.1d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v1f64.p0(<1 x double> %A, <1 x double> %B, <1 x double> %C, <1 x double> %D, ptr %addr)
  ret void
}

declare void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v8i16.p0(<8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v4f32.p0(<4 x float>, <4 x float>, <4 x float>, <4 x float>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, ptr) nounwind readonly
declare void @llvm.aarch64.neon.st1x4.v2f64.p0(<2 x double>, <2 x double>, <2 x double>, <2 x double>, ptr) nounwind readonly

define void @st1_x4_v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v16i8:
; CHECK: st1.16b { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, ptr %addr)
  ret void
}

define void @st1_x4_v8i16(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, <8 x i16> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v8i16:
; CHECK: st1.8h { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v8i16.p0(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, <8 x i16> %D, ptr %addr)
  ret void
}

define void @st1_x4_v4i32(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, <4 x i32> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v4i32:
; CHECK: st1.4s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v4i32.p0(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, <4 x i32> %D, ptr %addr)
  ret void
}

define void @st1_x4_v4f32(<4 x float> %A, <4 x float> %B, <4 x float> %C, <4 x float> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v4f32:
; CHECK: st1.4s { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v4f32.p0(<4 x float> %A, <4 x float> %B, <4 x float> %C, <4 x float> %D, ptr %addr)
  ret void
}

define void @st1_x4_v2i64(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v2i64:
; CHECK: st1.2d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %addr)
  ret void
}

define void @st1_x4_v2f64(<2 x double> %A, <2 x double> %B, <2 x double> %C, <2 x double> %D, ptr %addr) {
; CHECK-LABEL: st1_x4_v2f64:
; CHECK: st1.2d { {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} }, [x0]
  call void @llvm.aarch64.neon.st1x4.v2f64.p0(<2 x double> %A, <2 x double> %B, <2 x double> %C, <2 x double> %D, ptr %addr)
  ret void
}
