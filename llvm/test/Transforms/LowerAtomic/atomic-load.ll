; RUN: opt < %s -passes=lower-atomic -S | FileCheck %s

define i8 @add() {
; CHECK-LABEL: @add(
  %i = alloca i8
  %j = atomicrmw add ptr %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: add
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define i8 @nand() {
; CHECK-LABEL: @nand(
  %i = alloca i8
  %j = atomicrmw nand ptr %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: and
; CHECK-NEXT: xor
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define i8 @min() {
; CHECK-LABEL: @min(
  %i = alloca i8
  %j = atomicrmw min ptr %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define float @fadd() {
; CHECK-LABEL: @fadd(
  %i = alloca float
  %j = atomicrmw fadd ptr %i, float 42.0 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: fadd
; CHECK-NEXT: store
  ret float %j
; CHECK: ret float [[INST]]
}

define float @fsub() {
; CHECK-LABEL: @fsub(
  %i = alloca float
  %j = atomicrmw fsub ptr %i, float 42.0 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: fsub
; CHECK-NEXT: store
  ret float %j
; CHECK: ret float [[INST]]
}

define float @fmax() {
; CHECK-LABEL: @fmax(
  %i = alloca float
  %j = atomicrmw fmax ptr %i, float 42.0 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: call float @llvm.maxnum.f32
; CHECK-NEXT: store
  ret float %j
; CHECK: ret float [[INST]]
}

define float @fmin() {
; CHECK-LABEL: @fmin(
  %i = alloca float
  %j = atomicrmw fmin ptr %i, float 42.0 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: call float @llvm.minnum.f32
; CHECK-NEXT: store
  ret float %j
; CHECK: ret float [[INST]]
}

define float @fmaximum() {
; CHECK-LABEL: @fmaximum(
  %i = alloca float
  %j = atomicrmw fmaximum ptr %i, float 42.0 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: call float @llvm.maximum.f32
; CHECK-NEXT: store
  ret float %j
; CHECK: ret float [[INST]]
}

define float @fminimum() {
; CHECK-LABEL: @fminimum(
  %i = alloca float
  %j = atomicrmw fminimum ptr %i, float 42.0 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: call float @llvm.minimum.f32
; CHECK-NEXT: store
  ret float %j
; CHECK: ret float [[INST]]
}

define <2 x half> @fmaximum_2xhalf(<2 x half> %val) {
; CHECK-LABEL: @fmaximum_2xhalf(
  %i = alloca <2 x half>, align 4
  %j = atomicrmw fmaximum ptr %i, <2 x half> %val monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: call <2 x half> @llvm.maximum.v2f16
; CHECK-NEXT: store
  ret <2 x half> %j
}

define <2 x half> @fminimum_2xhalf(<2 x half> %val) {
; CHECK-LABEL: @fminimum_2xhalf(
  %i = alloca <2 x half>, align 4
  %j = atomicrmw fminimum ptr %i, <2 x half> %val monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: call <2 x half> @llvm.minimum.v2f16
; CHECK-NEXT: store
  ret <2 x half> %j
}

