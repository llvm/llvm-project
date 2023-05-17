// RUN: mlir-opt --convert-gpu-to-nvvm="use-opaque-pointers=0" --split-input-file %s | FileCheck %s
// RUN: mlir-opt --convert-gpu-to-nvvm="index-bitwidth=32 use-opaque-pointers=0" --split-input-file %s | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_load_op() ->
  // CHECK-SAME: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  // CHECK32-LABEL: func @gpu_wmma_load_op() ->
  func.func @gpu_wmma_load_op() -> (!gpu.mma_matrix<16x16xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index, transpose} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK:  %[[INX:.*]] = llvm.mlir.constant(16 : index) : i64
    // CHECK: %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[{{.*}}, {{.*}}]
    // CHECK:  %[[BASE:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK:  %[[LDM:.*]] = llvm.mlir.constant(32 : index) : i64
    // CHECK:  %[[LI:.*]] = llvm.mul %[[INX]], %[[LDM]]  : i64
    // CHECK:  %[[LIJ:.*]] = llvm.add %[[LI]], %[[INX]]  : i64
    // CHECK:  %[[ADDRESS:.*]] = llvm.getelementptr %[[BASE]][%[[LIJ]]] : (!llvm.ptr<f16, 3>, i64) -> !llvm.ptr<f16, 3>
    // CHECK:  %[[LDM32:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:  %[[FRAG:.*]] = nvvm.wmma.load %[[ADDRESS]], %[[LDM32]]
    // CHECK-SAME: {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<col>, m = 16 : i32, n = 16 : i32}  : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  llvm.return %[[FRAG]] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>

    // CHECK32:  %[[INX:.*]] = llvm.mlir.constant(16 : index) : i32
    // CHECK32: %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[{{.*}}, {{.*}}]
    // CHECK32:  %[[BASE:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i32, array<2 x i32>, array<2 x i32>)>
    // CHECK32:  %[[LDM:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK32:  %[[LI:.*]] = llvm.mul %[[INX]], %[[LDM]]  : i32
    // CHECK32:  %[[LIJ:.*]] = llvm.add %[[LI]], %[[INX]]  : i32
    // CHECK32:  %[[ADDRESS:.*]] = llvm.getelementptr %[[BASE]][%[[LIJ]]] : (!llvm.ptr<f16, 3>, i32) -> !llvm.ptr<f16, 3>
    // CHECK32:  %[[LDM32:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK32:  %[[FRAG:.*]] = nvvm.wmma.load %[[ADDRESS]], %[[LDM32]]
    // CHECK32-SAME: {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<col>, m = 16 : i32, n = 16 : i32}  : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK32:  llvm.return %[[FRAG]] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
  }
}
