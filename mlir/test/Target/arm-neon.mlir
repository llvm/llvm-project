// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: arm_neon_smull
llvm.func @arm_neon_smull(%arg0: vector<8xi8>, %arg1: vector<8xi8>) -> !llvm.struct<(vector<8xi16>, vector<4xi32>, vector<2xi64>)> {
  //      CHECK: %[[V0:.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %{{.*}}, <8 x i8> %{{.*}})
  // CHECK-NEXT: %[[V00:.*]] = shufflevector <8 x i16> %3, <8 x i16> %[[V0]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %0 = "llvm_arm_neon.smull"(%arg0, %arg1) : (vector<8xi8>, vector<8xi8>) -> vector<8xi16>
  %1 = llvm.shufflevector %0, %0 [3, 4, 5, 6] : vector<8xi16>, vector<8xi16>

  // CHECK-NEXT: %[[V1:.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %[[V00]], <4 x i16> %[[V00]])
  // CHECK-NEXT: %[[V11:.*]] = shufflevector <4 x i32> %[[V1]], <4 x i32> %[[V1]], <2 x i32> <i32 1, i32 2>
  %2 = "llvm_arm_neon.smull"(%1, %1) : (vector<4xi16>, vector<4xi16>) -> vector<4xi32>
  %3 = llvm.shufflevector %2, %2 [1, 2] : vector<4xi32>, vector<4xi32>

  // CHECK-NEXT: %[[V1:.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %[[V11]], <2 x i32> %[[V11]])
  %4 = "llvm_arm_neon.smull"(%3, %3) : (vector<2xi32>, vector<2xi32>) -> vector<2xi64>

  %5 = llvm.mlir.undef : !llvm.struct<(vector<8xi16>, vector<4xi32>, vector<2xi64>)>
  %6 = llvm.insertvalue %0, %5[0] : !llvm.struct<(vector<8xi16>, vector<4xi32>, vector<2xi64>)>
  %7 = llvm.insertvalue %2, %6[1] : !llvm.struct<(vector<8xi16>, vector<4xi32>, vector<2xi64>)>
  %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(vector<8xi16>, vector<4xi32>, vector<2xi64>)>

  //      CHECK: ret { <8 x i16>, <4 x i32>, <2 x i64> }
  llvm.return %8 : !llvm.struct<(vector<8xi16>, vector<4xi32>, vector<2xi64>)>
}
