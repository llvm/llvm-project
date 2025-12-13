// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1250 \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @wmma_k4
func.func @wmma_k4(%arg0 : vector<2xf32>, %arg1 : vector<8xf32>) {
  // CHECK: rocdl.wmma.f32.16x16x4.f32 %arg0, %arg0, %arg1
  amdgpu.wmma 16x16x4 %arg0 * %arg0 + %arg1 : vector<2xf32>, vector<2xf32>, vector<8xf32>
  return
}

// CHECK-LABEL: @wmma_k32
func.func @wmma_k32(%arg0 : vector<16xf16>, %arg1 : vector<16xbf16>, %arg2 : vector<8xf32>,
                    %arg3 : vector<8xf16>, %arg4 : vector<8xbf16>) {
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %arg0, %arg0, %arg2
  amdgpu.wmma 16x16x32 %arg0 * %arg0 + %arg2 : vector<16xf16>, vector<16xf16>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x32.f16 %arg0, %arg0, {{.*}} : (vector<16xf16>, vector<16xf16>, vector<8xf16>)
  amdgpu.wmma 16x16x32 %arg0 * %arg0 + %arg3 : vector<16xf16>, vector<16xf16>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x32.bf16 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x32 %arg1 * %arg1 + %arg2 : vector<16xbf16>, vector<16xbf16>, vector<8xf32>

  // CHECK: rocdl.wmma.bf16.16x16x32.bf16 {{.*}}, {{.*}}, {{.*}} : (vector<16xbf16>, vector<16xbf16>, vector<8xbf16>)
  amdgpu.wmma 16x16x32 %arg1 * %arg1 + %arg4 : vector<16xbf16>, vector<16xbf16>, vector<8xbf16>

  return
}

// CHECK-LABEL: @wmma_k64
func.func @wmma_k64(%arg0 : vector<32xi8>, %arg1 : vector<32xf8E4M3FN>, %arg2 : vector<32xf8E5M2>,
                    %arg3 : vector<8xi32>, %arg4 : vector<8xf32>, %arg5 : vector<8xf16>) {
  // CHECK: rocdl.wmma.i32.16x16x64.iu8 {{.*}}, {{.*}}, %arg3 {clamp = true, signA = true, signB = true}
  amdgpu.wmma 16x16x64 %arg0 * %arg0 + %arg3 {clamp} : vector<32xi8>, vector<32xi8>, vector<8xi32>

  // CHECK: rocdl.wmma.f32.16x16x64.fp8_fp8 {{.*}}, {{.*}}, %arg4
  amdgpu.wmma 16x16x64 %arg1 * %arg1 + %arg4 : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x64.fp8_fp8 {{.*}}, {{.*}}, %arg5 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x64 %arg1 * %arg1 + %arg5 : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x64.fp8_bf8 {{.*}}, {{.*}}, %arg4
  amdgpu.wmma 16x16x64 %arg1 * %arg2 + %arg4 : vector<32xf8E4M3FN>, vector<32xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x64.fp8_bf8 {{.*}}, {{.*}}, %arg5 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x64 %arg1 * %arg2 + %arg5 : vector<32xf8E4M3FN>, vector<32xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x64.bf8_bf8 {{.*}}, {{.*}}, %arg4
  amdgpu.wmma 16x16x64 %arg2 * %arg2 + %arg4 : vector<32xf8E5M2>, vector<32xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x64.bf8_bf8 {{.*}}, {{.*}}, %arg5 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x64 %arg2 * %arg2 + %arg5 : vector<32xf8E5M2>, vector<32xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x64.bf8_fp8 {{.*}}, {{.*}}, %arg4
  amdgpu.wmma 16x16x64 %arg2 * %arg1 + %arg4 : vector<32xf8E5M2>, vector<32xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x64.bf8_fp8 {{.*}}, {{.*}}, %arg5 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x64 %arg2 * %arg1 + %arg5 : vector<32xf8E5M2>, vector<32xf8E4M3FN>, vector<8xf16>

  return
}

// CHECK-LABEL: @wmma_k128
func.func @wmma_k128(%arg0 : vector<64xf8E4M3FN>, %arg1 : vector<64xf8E5M2>,
                     %arg2 : vector<8xf32>, %arg3 : vector<8xf16>) {
  // CHECK: rocdl.wmma.f32.16x16x128.fp8_fp8 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x128 %arg0 * %arg0 + %arg2 : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x128.fp8_fp8 {{.*}}, {{.*}}, %arg3 {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x128 %arg0 * %arg0 + %arg3 : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x128.fp8_bf8 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x128 %arg0 * %arg1 + %arg2 : vector<64xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x128.fp8_bf8 {{.*}}, {{.*}}, %arg3 {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x128 %arg0 * %arg1 + %arg3 : vector<64xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x128.bf8_bf8 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x128 %arg1 * %arg1 + %arg2 : vector<64xf8E5M2>, vector<64xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x128.bf8_bf8 {{.*}}, {{.*}}, %arg3 {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x128 %arg1 * %arg1 + %arg3 : vector<64xf8E5M2>, vector<64xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x128.bf8_fp8 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x128 %arg1 * %arg0 + %arg2 : vector<64xf8E5M2>, vector<64xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x128.bf8_fp8 {{.*}}, {{.*}}, %arg3 {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x128 %arg1 * %arg0 + %arg3 : vector<64xf8E5M2>, vector<64xf8E4M3FN>, vector<8xf16>

  return
}

// -----

func.func @wmma_unsupported_k(%arg0 : vector<8xf16>, %arg1 : vector<8xf32>) {
  // expected-error@below {{'amdgpu.wmma' op no intrinsic matching WMMA on the given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.wmma'}}
  amdgpu.wmma 16x16x16 %arg0 * %arg0 + %arg1 : vector<8xf16>, vector<8xf16>, vector<8xf32>
  return
}
