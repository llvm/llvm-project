// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1200 --allow-unregistered-dialect | FileCheck %s
func.func @mfma_to_rocdl(%arg0 : vector<8xf8E4M3FN>, %arg1 : vector<8xf8E5M2>,  %arg2 : vector<8xf32>) {
  // CHECK: rocdl.wmma.f32.16x16x16.fp8{{.*}}: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>
  amdgpu.wmma %arg0 * %arg0 + %arg2: vector<8xf8E4M3FN>, vector<8xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f32.16x16x16.bf8{{.*}}: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>
  amdgpu.wmma %arg1 * %arg1 + %arg2: vector<8xf8E5M2>, vector<8xf8E5M2>, vector<8xf32>
  func.return
}
