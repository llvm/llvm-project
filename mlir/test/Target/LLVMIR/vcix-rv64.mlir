// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s


// CHECK-LABEL: define <vscale x 4 x float> @binary_fv(<vscale x 4 x float> %0, float %1, i64 %2) {
// CHECK-NEXT:   %4 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.fv.se.nxv4f32.i64.nxv4f32.f32.i64(i64 1, <vscale x 4 x float> %0, float %1, i64 %2)
// CHECK-NEXT:   ret <vscale x 4 x float> %4
// CHECK-NEXT: }
llvm.func @binary_fv(%arg0: vector<[4]xf32>, %arg1: f32, %vl: i64) -> vector<[4]xf32> {
  %0 = "vcix.v.sv"(%arg0, %arg1, %vl) <{opcode = 1 : i64}> : (vector<[4]xf32>, f32, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// -----

// CHECK-LABEL: define <vscale x 4 x float> @binary_xv(<vscale x 4 x float> %0, i64 %1, i64 %2) {
// CHECK-NEXT:   %4 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.xv.se.nxv4f32.i64.nxv4f32.i64.i64(i64 3, <vscale x 4 x float> %0, i64 %1, i64 %2)
// CHECK-NEXT:   ret <vscale x 4 x float> %4
// CHECK-NEXT: }
llvm.func @binary_xv(%arg0: vector<[4]xf32>, %arg1: i64, %vl: i64) -> vector<[4]xf32> {
  %0 = "vcix.v.sv"(%arg0, %arg1, %vl) <{opcode = 3 : i64}> : (vector<[4]xf32>, i64, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// -----

// CHECK-LABEL: define <vscale x 4 x float> @binary_vv(<vscale x 4 x float> %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK-NEXT:   %4 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.vv.se.nxv4f32.i64.nxv4f32.nxv4f32.i64(i64 3, <vscale x 4 x float> %0, <vscale x 4 x float> %1, i64 %2)
// CHECK-NEXT:   ret <vscale x 4 x float> %4
// CHECK-NEXT: }
llvm.func @binary_vv(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %vl: i64) -> vector<[4]xf32> {
  %0 = "vcix.v.sv"(%arg0, %arg1, %vl) <{opcode = 3 : i64}> : (vector<[4]xf32>, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// -----

// CHECK-LABEL: define <vscale x 4 x float> @binary_iv(<vscale x 4 x float> %0, i64 %1) {
// CHECK-NEXT:   %3 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.iv.se.nxv4f32.i64.nxv4f32.i64.i64(i64 3, <vscale x 4 x float> %0, i64 5, i64 %1)
// CHECK-NEXT:   ret <vscale x 4 x float> %3
// CHECK-NEXT: }
llvm.func @binary_iv(%arg0: vector<[4]xf32>, %vl: i64) -> vector<[4]xf32> {
  %0 = "vcix.v.iv"(%arg0, %vl) <{opcode = 3 : i64, imm = 5 : i64}> : (vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// -----

// CHECK: define <4 x float> @binary_fixed_fv(<4 x float> %0, float %1) {
// CHECK-NEXT:   %3 = call <4 x float> @llvm.riscv.sf.vc.v.fv.se.v4f32.i64.v4f32.f32.i64(i64 1, <4 x float> %0, float %1, i64 4)
// CHECK-NEXT:   ret <4 x float> %3
// CHECK-NEXT: }
llvm.func @binary_fixed_fv(%arg0: vector<4xf32>, %arg1: f32) -> vector<4xf32> {
  %0 = "vcix.v.sv"(%arg0, %arg1) <{opcode = 1 : i64}> : (vector<4xf32>, f32) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: define <4 x float> @binary_fixed_xv(<4 x float> %0, i64 %1) {
// CHECK-NEXT:   %3 = call <4 x float> @llvm.riscv.sf.vc.v.xv.se.v4f32.i64.v4f32.i64.i64(i64 3, <4 x float> %0, i64 %1, i64 4)
// CHECK-NEXT:   ret <4 x float> %3
// CHECK-NEXT: }
llvm.func @binary_fixed_xv(%arg0: vector<4xf32>, %arg1: i64) -> vector<4xf32> {
  %0 = "vcix.v.sv"(%arg0, %arg1) <{opcode = 3 : i64}> : (vector<4xf32>, i64) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: define <4 x float> @binary_fixed_vv(<4 x float> %0, <4 x float> %1) {
// CHECK-NEXT:   %3 = call <4 x float> @llvm.riscv.sf.vc.v.vv.se.v4f32.i64.v4f32.v4f32.i64(i64 3, <4 x float> %0, <4 x float> %1, i64 4)
// CHECK-NEXT:   ret <4 x float> %3
// CHECK-NEXT: }
llvm.func @binary_fixed_vv(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = "vcix.v.sv"(%arg0, %arg1) <{opcode = 3 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: define <4 x float> @binary_fixed_iv(<4 x float> %0) {
// CHECK-NEXT:   %2 = call <4 x float> @llvm.riscv.sf.vc.v.iv.se.v4f32.i64.v4f32.i64.i64(i64 3, <4 x float> %0, i64 5, i64 4)
// CHECK-NEXT:   ret <4 x float> %2
// CHECK-NEXT: }
llvm.func @binary_fixed_iv(%arg0: vector<4xf32>) -> vector<4xf32> {
  %0 = "vcix.v.iv"(%arg0) <{opcode = 3 : i64, imm = 5 : i64}> : (vector<4xf32>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// Test integer type

// -----

// CHECK-LABEL: define <vscale x 4 x i32> @binary_i_fv(<vscale x 4 x i32> %0, float %1, i64 %2) {
// CHECK-NEXT:   %4 = call <vscale x 4 x i32> @llvm.riscv.sf.vc.v.fv.se.nxv4i32.i64.nxv4i32.f32.i64(i64 1, <vscale x 4 x i32> %0, float %1, i64 %2)
// CHECK-NEXT:   ret <vscale x 4 x i32> %4
// CHECK-NEXT: }
llvm.func @binary_i_fv(%arg0: vector<[4]xi32>, %arg1: f32, %vl: i64) -> vector<[4]xi32> {
  %0 = "vcix.v.sv"(%arg0, %arg1, %vl) <{opcode = 1 : i64}> : (vector<[4]xi32>, f32, i64) -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

// -----

// CHECK-LABEL: define <vscale x 4 x i32> @binary_i_xv(<vscale x 4 x i32> %0, i64 %1, i64 %2) {
// CHECK-NEXT:   %4 = call <vscale x 4 x i32> @llvm.riscv.sf.vc.v.xv.se.nxv4i32.i64.nxv4i32.i64.i64(i64 3, <vscale x 4 x i32> %0, i64 %1, i64 %2)
// CHECK-NEXT:   ret <vscale x 4 x i32> %4
// CHECK-NEXT: }
llvm.func @binary_i_xv(%arg0: vector<[4]xi32>, %arg1: i64, %vl: i64) -> vector<[4]xi32> {
  %0 = "vcix.v.sv"(%arg0, %arg1, %vl) <{opcode = 3 : i64}> : (vector<[4]xi32>, i64, i64) -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

// -----

// CHECK-LABEL: define <vscale x 4 x i32> @binary_i_vv(<vscale x 4 x i32> %0, <vscale x 4 x i32> %1, i64 %2) {
// CHECK-NEXT:   %4 = call <vscale x 4 x i32> @llvm.riscv.sf.vc.v.vv.se.nxv4i32.i64.nxv4i32.nxv4i32.i64(i64 3, <vscale x 4 x i32> %0, <vscale x 4 x i32> %1, i64 %2)
// CHECK-NEXT:   ret <vscale x 4 x i32> %4
// CHECK-NEXT: }
llvm.func @binary_i_vv(%arg0: vector<[4]xi32>, %arg1: vector<[4]xi32>, %vl: i64) -> vector<[4]xi32> {
  %0 = "vcix.v.sv"(%arg0, %arg1, %vl) <{opcode = 3 : i64}> : (vector<[4]xi32>, vector<[4]xi32>, i64) -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

// -----

// CHECK-LABEL: define <vscale x 4 x i32> @binary_i_iv(<vscale x 4 x i32> %0, i64 %1) {
// CHECK-NEXT:   %3 = call <vscale x 4 x i32> @llvm.riscv.sf.vc.v.iv.se.nxv4i32.i64.nxv4i32.i64.i64(i64 3, <vscale x 4 x i32> %0, i64 5, i64 %1)
// CHECK-NEXT:   ret <vscale x 4 x i32> %3
// CHECK-NEXT: }
llvm.func @binary_i_iv(%arg0: vector<[4]xi32>, %vl: i64) -> vector<[4]xi32> {
  %0 = "vcix.v.iv"(%arg0, %vl) <{opcode = 3 : i64, imm = 5 : i64}> : (vector<[4]xi32>, i64) -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

// -----

// CHECK: define <4 x i32> @binary_i_fixed_fv(<4 x i32> %0, float %1) {
// CHECK-NEXT:   %3 = call <4 x i32> @llvm.riscv.sf.vc.v.fv.se.v4i32.i64.v4i32.f32.i64(i64 1, <4 x i32> %0, float %1, i64 4)
// CHECK-NEXT:   ret <4 x i32> %3
// CHECK-NEXT: }
llvm.func @binary_i_fixed_fv(%arg0: vector<4xi32>, %arg1: f32) -> vector<4xi32> {
  %0 = "vcix.v.sv"(%arg0, %arg1) <{opcode = 1 : i64}> : (vector<4xi32>, f32) -> vector<4xi32>
  llvm.return %0 : vector<4xi32>
}

// -----

// CHECK-LABEL: define <4 x i32> @binary_i_fixed_xv(<4 x i32> %0, i64 %1) {
// CHECK-NEXT:   %3 = call <4 x i32> @llvm.riscv.sf.vc.v.xv.se.v4i32.i64.v4i32.i64.i64(i64 3, <4 x i32> %0, i64 %1, i64 4)
// CHECK-NEXT:   ret <4 x i32> %3
// CHECK-NEXT: }
llvm.func @binary_i_fixed_xv(%arg0: vector<4xi32>, %arg1: i64) -> vector<4xi32> {
  %0 = "vcix.v.sv"(%arg0, %arg1) <{opcode = 3 : i64}> : (vector<4xi32>, i64) -> vector<4xi32>
  llvm.return %0 : vector<4xi32>
}

// -----

// CHECK-LABEL: define <4 x i32> @binary_i_fixed_vv(<4 x i32> %0, <4 x i32> %1) {
// CHECK-NEXT:   %3 = call <4 x i32> @llvm.riscv.sf.vc.v.vv.se.v4i32.i64.v4i32.v4i32.i64(i64 3, <4 x i32> %0, <4 x i32> %1, i64 4)
// CHECK-NEXT:   ret <4 x i32> %3
// CHECK-NEXT: }
llvm.func @binary_i_fixed_vv(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {
  %0 = "vcix.v.sv"(%arg0, %arg1) <{opcode = 3 : i64}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
  llvm.return %0 : vector<4xi32>
}

// -----

// CHECK-LABEL: define <4 x i32> @binary_i_fixed_iv(<4 x i32> %0) {
// CHECK-NEXT:   %2 = call <4 x i32> @llvm.riscv.sf.vc.v.iv.se.v4i32.i64.v4i32.i64.i64(i64 3, <4 x i32> %0, i64 5, i64 4)
// CHECK-NEXT:   ret <4 x i32> %2
// CHECK-NEXT: }
llvm.func @binary_i_fixed_iv(%arg0: vector<4xi32>) -> vector<4xi32> {
  %0 = "vcix.v.iv"(%arg0) <{opcode = 3 : i64, imm = 5 : i64}> : (vector<4xi32>) -> vector<4xi32>
  llvm.return %0 : vector<4xi32>
}
