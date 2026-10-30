; Metal takes the delta of air.simd_shuffle_{up,down,xor} (and of the quad
; forms) from one lane for the whole SIMD-group. A relative shuffle whose delta
; is not uniform becomes air.simd_shuffle at the lane it reads (its own lane
; when that is outside the SIMD-group or quad); a uniform delta is kept.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

declare i32 @air.simd_shuffle_up.u.i32(i32, i16)
declare i32 @air.simd_shuffle_down.u.i32(i32, i16)
declare float @air.simd_shuffle_xor.f32(float, i16)
declare i32 @air.quad_shuffle_down.u.i32(i32, i16)

; CHECK-LABEL: define void @up
; CHECK-SAME: i32 [[LANE:%simdlane[0-9]*]])
; CHECK: [[D:%.*]] = and i32 %d32, 65535
; CHECK: [[S:%.*]] = sub i32 [[LANE]], [[D]]
; CHECK: [[OK:%.*]] = icmp uge i32 [[LANE]], [[D]]
; CHECK: [[I:%.*]] = select i1 [[OK]], i32 [[S]], i32 [[LANE]]
; CHECK: [[I16:%.*]] = trunc i32 [[I]] to i16
; CHECK: call i32 @air.simd_shuffle.u.i32(i32 %v, i16 [[I16]])
; CHECK-NOT: simd_shuffle_up
define void @up(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %d32 = load i32, ptr addrspace(1) %in
  %d = trunc i32 %d32 to i16
  %r = call i32 @air.simd_shuffle_up.u.i32(i32 %v, i16 %d)
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: define void @down
; CHECK: [[S:%.*]] = add i32 %simdlane{{[0-9]*}}, [[D:%.*]]
; CHECK: [[OK:%.*]] = icmp ult i32 [[S]], 32
; CHECK: call i32 @air.simd_shuffle.u.i32(
define void @down(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %d32 = load i32, ptr addrspace(1) %in
  %d = trunc i32 %d32 to i16
  %r = call i32 @air.simd_shuffle_down.u.i32(i32 %v, i16 %d)
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: define void @xor
; CHECK: [[S:%.*]] = xor i32 %simdlane{{[0-9]*}}, [[D:%.*]]
; CHECK: [[OK:%.*]] = icmp ult i32 [[S]], 32
; CHECK: call float @air.simd_shuffle.f32(
define void @xor(ptr addrspace(1) %in, ptr addrspace(1) %out, float %v) "air-kernel" {
  %d32 = load i32, ptr addrspace(1) %in
  %d = trunc i32 %d32 to i16
  %r = call float @air.simd_shuffle_xor.f32(float %v, i16 %d)
  store float %r, ptr addrspace(1) %out
  ret void
}

; Within the quad: the quad lane is the SIMD lane's low two bits.
; CHECK-LABEL: define void @quad_down
; CHECK: [[Q:%.*]] = and i32 %simdlane{{[0-9]*}}, 3
; CHECK: [[S:%.*]] = add i32 [[Q]],
; CHECK: icmp ult i32 [[S]], 4
; CHECK: call i32 @air.quad_shuffle.u.i32(
define void @quad_down(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %d32 = load i32, ptr addrspace(1) %in
  %d = trunc i32 %d32 to i16
  %r = call i32 @air.quad_shuffle_down.u.i32(i32 %v, i16 %d)
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; A uniform delta (a constant, a constant-buffer load) stays a relative shuffle.
; CHECK-LABEL: define void @uniform
; CHECK: call i32 @air.simd_shuffle_up.u.i32(i32 %v, i16 1)
; CHECK: call i32 @air.simd_shuffle_down.u.i32(i32 %v, i16 %d)
define void @uniform(ptr addrspace(2) %c, ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %a = call i32 @air.simd_shuffle_up.u.i32(i32 %v, i16 1)
  %d32 = load i32, ptr addrspace(2) %c
  %d = trunc i32 %d32 to i16
  %b = call i32 @air.simd_shuffle_down.u.i32(i32 %v, i16 %d)
  %s = add i32 %a, %b
  store i32 %s, ptr addrspace(1) %out
  ret void
}
