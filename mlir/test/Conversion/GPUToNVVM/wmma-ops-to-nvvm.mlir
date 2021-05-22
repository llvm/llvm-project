// RUN: mlir-opt --convert-gpu-to-nvvm="index-bitwidth=32" --split-input-file %s | FileCheck %s

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_load_op() ->
  // CHECK-SAME: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)> {
  func @gpu_wmma_load_op() -> (!gpu.mma_matrix<16x16xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = constant 16 : index
    %j = constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK:  %[[INX:.*]] = llvm.mlir.constant(16 : index) : i32
    // CHECK: %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[{{.*}}, {{.*}}]
    // CHECK:  %[[LDM:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:  %[[LI:.*]] = llvm.mul %[[LDM]], %[[INX]]  : i32
    // CHECK:  %[[LIJ:.*]] = llvm.add %[[LI]], %[[INX]]  : i32
    // CHECK:  %[[OFFSET:.*]] = llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i32, array<2 x i32>, array<2 x i32>)>
    // CHECK:  %[[LIJO:.*]] = llvm.add %[[LIJ]], %[[OFFSET]]  : i32
    // CHECK:  %[[BASE:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i32, array<2 x i32>, array<2 x i32>)>
    // CHECK:  %[[ADDRESS:.*]] = llvm.getelementptr %[[BASE]][%[[LIJO]]] : (!llvm.ptr<f16, 3>, i32) -> !llvm.ptr<f16, 3>
    // CHECK:  %[[CADDRESS:.*]] = llvm.bitcast %[[ADDRESS]] : !llvm.ptr<f16, 3> to !llvm.ptr<i32, 3>
    // CHECK:  %[[FRAG:.*]] = nvvm.wmma.m16n16k16.load.a.f16.row.stride %[[CADDRESS]], %[[LDM]] : (!llvm.ptr<i32, 3>, i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  llvm.return %[[FRAG]] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
  }
}

// -----

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_store_op
  // CHECK-SAME: (%[[D:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>) {
  func @gpu_wmma_store_op(%arg0 : !gpu.mma_matrix<16x16xf16, "DOp">) -> () {
    %sg = memref.alloca(){alignment = 32} : memref<32x32xf16, 3>
    %i = constant 16 : index
    %j = constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %sg[%i,%j] {leadDimension= 32 : index} : !gpu.mma_matrix<16x16xf16, "DOp">, memref<32x32xf16, 3>
    // CHECK:  %[[INX:.*]] = llvm.mlir.constant(16 : index) : i32
    // CHECK: %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[{{.*}}, {{.*}}]
    // CHECK:  %[[LDM:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:  %[[LI:.*]] = llvm.mul %[[LDM]], %[[INX]]  : i32
    // CHECK:  %[[LIJ:.*]] = llvm.add %[[LI]], %[[INX]]  : i32
    // CHECK:  %[[OFFSET:.*]] = llvm.extractvalue %17[2] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i32, array<2 x i32>, array<2 x i32>)>
    // CHECK:  %[[LIJO:.*]] = llvm.add %[[LIJ]], %[[OFFSET]]  : i32
    // CHECK:  %[[BASE:.*]] = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i32, array<2 x i32>, array<2 x i32>)>
    // CHECK:  %[[ADDRESS:.*]] = llvm.getelementptr %[[BASE]][%[[LIJO]]] : (!llvm.ptr<f16, 3>, i32) -> !llvm.ptr<f16, 3>
    // CHECK:  %[[CADDRESS:.*]] = llvm.bitcast %[[ADDRESS]] : !llvm.ptr<f16, 3> to !llvm.ptr<i32, 3>
    // CHECK:  %[[EL1:.*]] = llvm.extractvalue %[[D]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[EL2:.*]] = llvm.extractvalue %[[D]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[EL3:.*]] = llvm.extractvalue %[[D]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[EL4:.*]] = llvm.extractvalue %[[D]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  nvvm.wmma.m16n16k16.store.d.f16.row.stride %[[CADDRESS]], %[[EL1]], %[[EL2]], %[[EL3]], %[[EL4]], %[[LDM]] : !llvm.ptr<i32, 3>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, i32
    // CHECK:  llvm.return
    return
  }
}

// -----

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_mma_op
  // CHECK-SAME: (%[[A:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>, %[[B:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>, %[[C:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>) {
  func @gpu_wmma_mma_op(%A : !gpu.mma_matrix<16x16xf16, "AOp">, %B : !gpu.mma_matrix<16x16xf16, "BOp">, %C : !gpu.mma_matrix<16x16xf16, "COp">) -> () {
    %D = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">, !gpu.mma_matrix<16x16xf16, "COp"> -> !gpu.mma_matrix<16x16xf16, "DOp">
    // CHECK:  %[[A1:.*]] = llvm.extractvalue %[[A]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[A2:.*]] = llvm.extractvalue %[[A]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[A3:.*]] = llvm.extractvalue %[[A]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[A4:.*]] = llvm.extractvalue %[[A]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[A5:.*]] = llvm.extractvalue %[[A]][4 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[A6:.*]] = llvm.extractvalue %[[A]][5 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[A7:.*]] = llvm.extractvalue %[[A]][6 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[A8:.*]] = llvm.extractvalue %[[A]][7 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[B1:.*]] = llvm.extractvalue %[[B]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[B2:.*]] = llvm.extractvalue %[[B]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[B3:.*]] = llvm.extractvalue %[[B]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[B4:.*]] = llvm.extractvalue %[[B]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[B5:.*]] = llvm.extractvalue %[[B]][4 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[B6:.*]] = llvm.extractvalue %[[B]][5 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[B7:.*]] = llvm.extractvalue %[[B]][6 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[B8:.*]] = llvm.extractvalue %[[B]][7 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[C1:.*]] = llvm.extractvalue %[[C]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[C2:.*]] = llvm.extractvalue %[[C]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[C3:.*]] = llvm.extractvalue %[[C]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[C4:.*]] = llvm.extractvalue %[[C]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %{{.*}} = nvvm.wmma.m16n16k16.mma.row.row.f16.f16 %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[A8]], %[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[B8]], %[[C1]], %[[C2]], %[[C3]], %[[C4]] : vector<2xf16> -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  llvm.return
    return
  }
}
