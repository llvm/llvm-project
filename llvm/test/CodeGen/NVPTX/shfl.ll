; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_30 | %ptxas-verify %if !ptxas-11.0 %{-arch=sm_30%} %}

declare i32 @llvm.nvvm.shfl.down.i32(i32, i32, i32)
declare float @llvm.nvvm.shfl.down.f32(float, i32, i32)
declare i32 @llvm.nvvm.shfl.up.i32(i32, i32, i32)
declare float @llvm.nvvm.shfl.up.f32(float, i32, i32)
declare i32 @llvm.nvvm.shfl.bfly.i32(i32, i32, i32)
declare float @llvm.nvvm.shfl.bfly.f32(float, i32, i32)
declare i32 @llvm.nvvm.shfl.idx.i32(i32, i32, i32)
declare float @llvm.nvvm.shfl.idx.f32(float, i32, i32)

; Try all four permutations of register and immediate parameters with
; shfl.down.

; CHECK-LABEL: .func{{.*}}shfl_down1
define i32 @shfl_down1(i32 %in) {
  ; CHECK: ld.param.u32 [[IN:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]], [[IN]], 1, 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.down.i32(i32 %in, i32 1, i32 2)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl_down2
define i32 @shfl_down2(i32 %in, i32 %width) {
  ; CHECK: ld.param.u32 [[IN1:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[IN2:%r[0-9]+]]
  ; CHECK: shfl.down.{{.}}32 %r{{[0-9]+}}, [[IN1]], [[IN2]], 3;
  %val = call i32 @llvm.nvvm.shfl.down.i32(i32 %in, i32 %width, i32 3)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl_down3
define i32 @shfl_down3(i32 %in, i32 %mask) {
  ; CHECK: ld.param.u32 [[IN1:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[IN2:%r[0-9]+]]
  ; CHECK: shfl.down.{{.}}32 %r{{[0-9]+}}, [[IN1]], 4, [[IN2]];
  %val = call i32 @llvm.nvvm.shfl.down.i32(i32 %in, i32 4, i32 %mask)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl_down4
define i32 @shfl_down4(i32 %in, i32 %width, i32 %mask) {
  ; CHECK: ld.param.u32 [[IN1:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[IN2:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[IN3:%r[0-9]+]]
  ; CHECK: shfl.down.{{.}}32 %r{{[0-9]+}}, [[IN1]], [[IN2]], [[IN3]];
  %val = call i32 @llvm.nvvm.shfl.down.i32(i32 %in, i32 %width, i32 %mask)
  ret i32 %val
}

; Try shfl.down with floating-point params.
; CHECK-LABEL: .func{{.*}}shfl_down_float
define float @shfl_down_float(float %in) {
  ; CHECK: ld.param.f32 [[IN:%f[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]], [[IN]], 5, 6;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %out = call float @llvm.nvvm.shfl.down.f32(float %in, i32 5, i32 6)
  ret float %out
}

; Try the rest of the shfl modes.  Hopefully they're declared in such a way
; that if shfl.down works correctly, they also work correctly.
define void @shfl_rest(i32 %in_i32, float %in_float, i32* %out_i32, float* %out_float) {
  ; CHECK: shfl.up.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, 1, 2;
  %up_i32 = call i32 @llvm.nvvm.shfl.up.i32(i32 %in_i32, i32 1, i32 2)
  store i32 %up_i32, i32* %out_i32

  ; CHECK: shfl.up.b32 %f{{[0-9]+}}, %f{{[0-9]+}}, 3, 4;
  %up_float = call float @llvm.nvvm.shfl.up.f32(float %in_float, i32 3, i32 4)
  store float %up_float, float* %out_float

  ; CHECK: shfl.bfly.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, 5, 6;
  %bfly_i32 = call i32 @llvm.nvvm.shfl.bfly.i32(i32 %in_i32, i32 5, i32 6)
  store i32 %bfly_i32, i32* %out_i32

  ; CHECK: shfl.bfly.b32 %f{{[0-9]+}}, %f{{[0-9]+}}, 7, 8;
  %bfly_float = call float @llvm.nvvm.shfl.bfly.f32(float %in_float, i32 7, i32 8)
  store float %bfly_float, float* %out_float

  ; CHECK: shfl.idx.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, 9, 10;
  %idx_i32 = call i32 @llvm.nvvm.shfl.idx.i32(i32 %in_i32, i32 9, i32 10)
  store i32 %idx_i32, i32* %out_i32

  ; CHECK: shfl.idx.b32 %f{{[0-9]+}}, %f{{[0-9]+}}, 11, 12;
  %idx_float = call float @llvm.nvvm.shfl.idx.f32(float %in_float, i32 11, i32 12)
  store float %idx_float, float* %out_float

  ret void
}
