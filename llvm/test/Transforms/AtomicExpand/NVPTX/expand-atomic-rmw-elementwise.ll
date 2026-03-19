; RUN: opt -S %s -passes=atomic-expand -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

define <4 x float> @preserve_fadd_v4f32_monotonic(ptr %addr, <4 x float> %val) {
; CHECK-LABEL: define <4 x float> @preserve_fadd_v4f32_monotonic(
; CHECK-SAME: ptr [[ADDR:%.*]], <4 x float> [[VAL:%.*]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[OLD:%.*]] = atomicrmw elementwise fadd ptr [[ADDR]], <4 x float> [[VAL]] monotonic, align 16
; CHECK-NEXT:   ret <4 x float> [[OLD]]
entry:
  %old = atomicrmw elementwise fadd ptr %addr, <4 x float> %val monotonic, align 16
  ret <4 x float> %old
}

define <4 x float> @preserve_fadd_v4f32_seq_cst(ptr %addr, <4 x float> %val) {
; CHECK-LABEL: define <4 x float> @preserve_fadd_v4f32_seq_cst(
; CHECK-SAME: ptr [[ADDR:%.*]], <4 x float> [[VAL:%.*]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   [[OLD:%.*]] = atomicrmw elementwise fadd ptr [[ADDR]], <4 x float> [[VAL]] monotonic, align 16
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   ret <4 x float> [[OLD]]
entry:
  %old = atomicrmw elementwise fadd ptr %addr, <4 x float> %val seq_cst, align 16
  ret <4 x float> %old
}

define <4 x i32> @expand_add_v4i32(ptr %addr, <4 x i32> %val) {
; CHECK-LABEL: define <4 x i32> @expand_add_v4i32(
; CHECK-SAME: ptr [[ADDR:%.*]], <4 x i32> [[VAL:%.*]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[V0:%.*]] = extractelement <4 x i32> [[VAL]], i64 0
; CHECK-NEXT:   [[O0:%.*]] = atomicrmw add ptr [[ADDR]], i32 [[V0]] monotonic, align 16
; CHECK-NEXT:   [[R0:%.*]] = insertelement <4 x i32> poison, i32 [[O0]], i64 0
; CHECK-NEXT:   [[P1:%.*]] = getelementptr inbounds <4 x i32>, ptr [[ADDR]], i64 0, i64 1
; CHECK-NEXT:   [[V1:%.*]] = extractelement <4 x i32> [[VAL]], i64 1
; CHECK-NEXT:   [[O1:%.*]] = atomicrmw add ptr [[P1]], i32 [[V1]] monotonic, align 4
; CHECK-NEXT:   [[R1:%.*]] = insertelement <4 x i32> [[R0]], i32 [[O1]], i64 1
; CHECK-NEXT:   [[P2:%.*]] = getelementptr inbounds <4 x i32>, ptr [[ADDR]], i64 0, i64 2
; CHECK-NEXT:   [[V2:%.*]] = extractelement <4 x i32> [[VAL]], i64 2
; CHECK-NEXT:   [[O2:%.*]] = atomicrmw add ptr [[P2]], i32 [[V2]] monotonic, align 8
; CHECK-NEXT:   [[R2:%.*]] = insertelement <4 x i32> [[R1]], i32 [[O2]], i64 2
; CHECK-NEXT:   [[P3:%.*]] = getelementptr inbounds <4 x i32>, ptr [[ADDR]], i64 0, i64 3
; CHECK-NEXT:   [[V3:%.*]] = extractelement <4 x i32> [[VAL]], i64 3
; CHECK-NEXT:   [[O3:%.*]] = atomicrmw add ptr [[P3]], i32 [[V3]] monotonic, align 4
; CHECK-NEXT:   [[R3:%.*]] = insertelement <4 x i32> [[R2]], i32 [[O3]], i64 3
; CHECK-NEXT:   ret <4 x i32> [[R3]]
entry:
  %old = atomicrmw elementwise add ptr %addr, <4 x i32> %val monotonic, align 16
  ret <4 x i32> %old
}
