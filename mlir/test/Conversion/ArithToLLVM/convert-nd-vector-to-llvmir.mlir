// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-llvm))" %s -split-input-file | FileCheck %s

// CHECK-LABEL: @vec_bin
func.func @vec_bin(%arg0: vector<2x2x2xf32>) -> vector<2x2x2xf32> {
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
func.func @sexti_vector(%arg0 : vector<1x2x3xi32>, %arg1 : vector<1x2x3xi64>) {
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
func.func @zexti_vector(%arg0 : vector<1x2x3xi32>, %arg1 : vector<1x2x3xi64>) {
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
func.func @sitofp_vector(%arg0 : vector<1x2x3xi32>) -> vector<1x2x3xf32> {
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
func.func @uitofp_vector(%arg0 : vector<1x2x3xi32>) -> vector<1x2x3xf32> {
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
func.func @fptosi_vector(%arg0 : vector<1x2x3xf32>) -> vector<1x2x3xi32> {
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
func.func @fptoui_vector(%arg0 : vector<1x2x3xf32>) -> vector<1x2x3xi32> {
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
func.func @fpext_vector(%arg0 : vector<1x2x3xf16>) -> vector<1x2x3xf64> {
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
func.func @fptrunc_vector(%arg0 : vector<1x2x3xf64>) -> vector<1x2x3xf16> {
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
func.func @trunci_vector(%arg0 : vector<1x2x3xi64>) -> vector<1x2x3xi16> {
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
func.func @shl_vector(%arg0 : vector<1x2x3xi64>) -> vector<1x2x3xi64> {
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
func.func @shrs_vector(%arg0 : vector<1x2x3xi64>) -> vector<1x2x3xi64> {
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
func.func @shru_vector(%arg0 : vector<1x2x3xi64>) -> vector<1x2x3xi64> {
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
func.func @bitcast_2d(%arg0: vector<2x4xf32>) {
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
func.func @select_2d(%arg0 : vector<4x3xi1>, %arg1 : vector<4x3xi32>, %arg2 : vector<4x3xi32>) {
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

// CHECK-LABEL: func @index_cast_2d(
// CHECK-SAME:    %[[ARG0:.*]]: vector<1x2x3xi1>)
func.func @index_cast_2d(%arg0: vector<1x2x3xi1>) {
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: %[[EXTRACT1:.*]] = llvm.extractvalue %[[SRC]][0, 0] : !llvm.array<1 x array<2 x vector<3xi1>>>
  // CHECK: %[[SEXT1:.*]] = llvm.sext %[[EXTRACT1]] : vector<3xi1> to vector<3xi{{.*}}>
  // CHECK: %[[INSERT1:.*]] = llvm.insertvalue %[[SEXT1]], %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi{{.*}}>>>
  // CHECK: %[[EXTRACT2:.*]] = llvm.extractvalue %[[SRC]][0, 1] : !llvm.array<1 x array<2 x vector<3xi1>>>
  // CHECK: %[[SEXT2:.*]] = llvm.sext %[[EXTRACT2]] : vector<3xi1> to vector<3xi{{.*}}>
  // CHECK: %[[INSERT2:.*]] = llvm.insertvalue %[[SEXT2]], %[[INSERT1]][0, 1] : !llvm.array<1 x array<2 x vector<3xi{{.*}}>>>
  %0 = arith.index_cast %arg0: vector<1x2x3xi1> to vector<1x2x3xindex>
  // CHECK: %[[EXTRACT3:.*]] = llvm.extractvalue %[[INSERT2]][0, 0] : !llvm.array<1 x array<2 x vector<3xi{{.*}}>>>
  // CHECK: %[[TRUNC1:.*]] = llvm.trunc %[[EXTRACT3]] : vector<3xi{{.*}}> to vector<3xi1>
  // CHECK: %[[INSERT3:.*]] = llvm.insertvalue %[[TRUNC1]], %{{.*}}[0, 0] : !llvm.array<1 x array<2 x vector<3xi1>>>
  // CHECK: %[[EXTRACT4:.*]] = llvm.extractvalue %[[INSERT2]][0, 1] : !llvm.array<1 x array<2 x vector<3xi{{.*}}>>>
  // CHECK: %[[TRUNC2:.*]] = llvm.trunc %[[EXTRACT4]] : vector<3xi{{.*}}> to vector<3xi1>
  // CHECK: %[[INSERT4:.*]] = llvm.insertvalue %[[TRUNC2]], %[[INSERT3]][0, 1] : !llvm.array<1 x array<2 x vector<3xi1>>>
  %1 = arith.index_cast %0: vector<1x2x3xindex> to vector<1x2x3xi1>
  return
}
