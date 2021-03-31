// RUN: mlir-opt -convert-std-to-llvm %s -split-input-file | FileCheck %s

// CHECK-LABEL: @vec_bin
func @vec_bin(%arg0: vector<2x2x2xf32>) -> vector<2x2x2xf32> {
  %0 = addf %arg0, %arg0 : vector<2x2x2xf32>
  return %0 : vector<2x2x2xf32>

//  CHECK-NEXT: llvm.mlir.undef : !llvm.array<2 x array<2 x vector<2xf32>>>

// This block appears 2x2 times
//  CHECK-NEXT: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>
//  CHECK-NEXT: llvm.extractvalue %{{.*}}[0, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>
//  CHECK-NEXT: llvm.fadd %{{.*}} : vector<2xf32>
//  CHECK-NEXT: llvm.insertvalue %{{.*}}[0, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>

// We check the proper indexing of extract/insert in the remaining 3 positions.
//       CHECK: llvm.extractvalue %{{.*}}[0, 1] : !llvm.array<2 x array<2 x vector<2xf32>>>
//       CHECK: llvm.insertvalue %{{.*}}[0, 1] : !llvm.array<2 x array<2 x vector<2xf32>>>
//       CHECK: llvm.extractvalue %{{.*}}[1, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>
//       CHECK: llvm.insertvalue %{{.*}}[1, 0] : !llvm.array<2 x array<2 x vector<2xf32>>>
//       CHECK: llvm.extractvalue %{{.*}}[1, 1] : !llvm.array<2 x array<2 x vector<2xf32>>>
//       CHECK: llvm.insertvalue %{{.*}}[1, 1] : !llvm.array<2 x array<2 x vector<2xf32>>>
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
  %0 = sexti %arg0: vector<1x2x3xi32> to vector<1x2x3xi64>
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
  %0 = zexti %arg0: vector<1x2x3xi32> to vector<1x2x3xi64>
  return
}
