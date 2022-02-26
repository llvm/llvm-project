// RUN: mlir-opt -convert-arith-to-llvm %s -split-input-file | FileCheck %s

// CHECK-LABEL: @vec_bin
func @vec_bin(%arg0: vector<2x2x2xf32>) -> vector<2x2x2xf32> {
  // CHECK: llvm.mlir.undef : !llvm.array<2 x array<2 x vector<2xf32>>>

  // This block appears 2x2 times
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>
  // CHECK-NEXT: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>
  // CHECK-NEXT: llvm.fadd %{{.*}} : vector<2xf32>
  // CHECK-NEXT: llvm.insertvalue %{{.*}}[0, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>

  // We check the proper indexing of extract/insert in the remaining 3 positions.
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<2 x array<2 x vector<2xf32>>>
  // CHECK: llvm.insertvalue %{{.*}}[0, 1] : !llvm.array<2 x array<2 x vector<2xf32>>>
  // CHECK: llvm.extractvalue %{{.*}}[1, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>
  // CHECK: llvm.insertvalue %{{.*}}[1, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>
  // CHECK: llvm.extractvalue %{{.*}}[1, 1] : !llvm.array<2 x array<2 x vector<2xf32>>>
  // CHECK: llvm.insertvalue %{{.*}}[1, 1] : !llvm.array<2 x array<2 x vector<2xf32>>>
  %0 = arith.addf %arg0, %arg0 : vector<2x2x2xf32>
  return %0 : vector<2x2x2xf32>
}

// CHECK-LABEL: @sexti
func @sexti_vector(%arg0 : vector<1x2x3xi32>, %arg1 : vector<1x2x3xi64>) {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.sext %{{.*}} : vector<3xi32> to vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.sext %{{.*}} : vector<3xi32> to vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  %0 = arith.extsi %arg0: vector<1x2x3xi32> to vector<1x2x3xi64>
  return
}

// CHECK-LABEL: @zexti
func @zexti_vector(%arg0 : vector<1x2x3xi32>, %arg1 : vector<1x2x3xi64>) {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.zext %{{.*}} : vector<3xi32> to vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.zext %{{.*}} : vector<3xi32> to vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  %0 = arith.extui %arg0: vector<1x2x3xi32> to vector<1x2x3xi64>
  return
}

// CHECK-LABEL: @sitofp
func @sitofp_vector(%arg0 : vector<1x2x3xi32>) -> vector<1x2x3xf32> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xf32>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.sitofp %{{.*}} : vector<3xi32> to vector<3xf32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xf32>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.sitofp %{{.*}} : vector<3xi32> to vector<3xf32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xf32>>>
  %0 = arith.sitofp %arg0: vector<1x2x3xi32> to vector<1x2x3xf32>
  return %0 : vector<1x2x3xf32>
}

// CHECK-LABEL: @uitofp
func @uitofp_vector(%arg0 : vector<1x2x3xi32>) -> vector<1x2x3xf32> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xf32>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.uitofp %{{.*}} : vector<3xi32> to vector<3xf32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xf32>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.uitofp %{{.*}} : vector<3xi32> to vector<3xf32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xf32>>>
  %0 = arith.uitofp %arg0: vector<1x2x3xi32> to vector<1x2x3xf32>
  return %0 : vector<1x2x3xf32>
}

// CHECK-LABEL: @fptosi
func @fptosi_vector(%arg0 : vector<1x2x3xf32>) -> vector<1x2x3xi32> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xf32>>>
  // CHECK: llvm.fptosi %{{.*}} : vector<3xf32> to vector<3xi32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xf32>>>
  // CHECK: llvm.fptosi %{{.*}} : vector<3xf32> to vector<3xi32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi32>>>
  %0 = arith.fptosi %arg0: vector<1x2x3xf32> to vector<1x2x3xi32>
  return %0 : vector<1x2x3xi32>
}

// CHECK-LABEL: @fptoui
func @fptoui_vector(%arg0 : vector<1x2x3xf32>) -> vector<1x2x3xi32> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xf32>>>
  // CHECK: llvm.fptoui %{{.*}} : vector<3xf32> to vector<3xi32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi32>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xf32>>>
  // CHECK: llvm.fptoui %{{.*}} : vector<3xf32> to vector<3xi32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi32>>>
  %0 = arith.fptoui %arg0: vector<1x2x3xf32> to vector<1x2x3xi32>
  return %0 : vector<1x2x3xi32>
}

// CHECK-LABEL: @fpext
func @fpext_vector(%arg0 : vector<1x2x3xf16>) -> vector<1x2x3xf64> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xf64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xf16>>>
  // CHECK: llvm.fpext %{{.*}} : vector<3xf16> to vector<3xf64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xf64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xf16>>>
  // CHECK: llvm.fpext %{{.*}} : vector<3xf16> to vector<3xf64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xf64>>>
  %0 = arith.extf %arg0: vector<1x2x3xf16> to vector<1x2x3xf64>
  return %0 : vector<1x2x3xf64>
}

// CHECK-LABEL: @fptrunc
func @fptrunc_vector(%arg0 : vector<1x2x3xf64>) -> vector<1x2x3xf16> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xf16>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xf64>>>
  // CHECK: llvm.fptrunc %{{.*}} : vector<3xf64> to vector<3xf16>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xf16>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xf64>>>
  // CHECK: llvm.fptrunc %{{.*}} : vector<3xf64> to vector<3xf16>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xf16>>>
  %0 = arith.truncf %arg0: vector<1x2x3xf64> to vector<1x2x3xf16>
  return %0 : vector<1x2x3xf16>
}

// CHECK-LABEL: @trunci
func @trunci_vector(%arg0 : vector<1x2x3xi64>) -> vector<1x2x3xi16> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xi16>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.trunc %{{.*}} : vector<3xi64> to vector<3xi16>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi16>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.trunc %{{.*}} : vector<3xi64> to vector<3xi16>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi16>>>
  %0 = arith.trunci %arg0: vector<1x2x3xi64> to vector<1x2x3xi16>
  return %0 : vector<1x2x3xi16>
}

// CHECK-LABEL: @shl
func @shl_vector(%arg0 : vector<1x2x3xi64>) -> vector<1x2x3xi64> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.shl %{{.*}}, %{{.*}} : vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.shl %{{.*}}, %{{.*}} : vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  %c1 = arith.constant dense<1> : vector<1x2x3xi64>
  %0 = arith.shli %arg0, %c1 : vector<1x2x3xi64>
  return %0 : vector<1x2x3xi64>
}

// CHECK-LABEL: @shrs
func @shrs_vector(%arg0 : vector<1x2x3xi64>) -> vector<1x2x3xi64> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  %c1 = arith.constant dense<1> : vector<1x2x3xi64>
  %0 = arith.shrsi %arg0, %c1 : vector<1x2x3xi64>
  return %0 : vector<1x2x3xi64>
}

// CHECK-LABEL: @shru
func @shru_vector(%arg0 : vector<1x2x3xi64>) -> vector<1x2x3xi64> {
  // CHECK: llvm.mlir.undef : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : vector<3xi64>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0, 1] : !llvm.array<1 x array<2 x vector<3xi64>>>
  %c1 = arith.constant dense<1> : vector<1x2x3xi64>
  %0 = arith.shrui %arg0, %c1 : vector<1x2x3xi64>
  return %0 : vector<1x2x3xi64>
}

// -----

// CHECK-LABEL: @bitcast_2d
func @bitcast_2d(%arg0: vector<2x4xf32>) {
  // CHECK: llvm.mlir.undef
  // CHECK: llvm.extractvalue %{{.*}}[0]
  // CHECK: llvm.bitcast %{{.*}} : vector<4xf32> to vector<4xi32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0]
  // CHECK: llvm.extractvalue %{{.*}}[1]
  // CHECK: llvm.bitcast %{{.*}} : vector<4xf32> to vector<4xi32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1]
  arith.bitcast %arg0 : vector<2x4xf32> to vector<2x4xi32>
  return
}

// -----

// CHECK-LABEL: func @select_2d(
func @select_2d(%arg0 : vector<4x3xi1>, %arg1 : vector<4x3xi32>, %arg2 : vector<4x3xi32>) {
  // CHECK: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: %[[ARG2:.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: %[[EXTRACT1:.*]] = llvm.extractvalue %[[ARG0]][0] : !llvm.array<4 x vector<3xi1>>
  // CHECK: %[[EXTRACT2:.*]] = llvm.extractvalue %[[ARG1]][0] : !llvm.array<4 x vector<3xi32>>
  // CHECK: %[[EXTRACT3:.*]] = llvm.extractvalue %[[ARG2]][0] : !llvm.array<4 x vector<3xi32>>
  // CHECK: %[[SELECT:.*]] = llvm.select %[[EXTRACT1]], %[[EXTRACT2]], %[[EXTRACT3]] : vector<3xi1>, vector<3xi32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[SELECT]], %{{.*}}[0] : !llvm.array<4 x vector<3xi32>>
  %0 = arith.select %arg0, %arg1, %arg2 : vector<4x3xi1>, vector<4x3xi32>
  func.return
}
