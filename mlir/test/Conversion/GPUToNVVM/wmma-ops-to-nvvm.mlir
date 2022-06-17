// RUN: mlir-opt --convert-gpu-to-nvvm --split-input-file %s | FileCheck %s
// RUN: mlir-opt --convert-gpu-to-nvvm="index-bitwidth=32" --split-input-file %s | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_load_op() ->
  // CHECK-SAME: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  // CHECK32-LABEL: func @gpu_wmma_load_op() ->
  func.func @gpu_wmma_load_op() -> (!gpu.mma_matrix<16x16xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK:  %[[INX:.*]] = llvm.mlir.constant(16 : index) : i64
    // CHECK: %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[{{.*}}, {{.*}}]
    // CHECK:  %[[BASE:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK:  %[[LDM:.*]] = llvm.mlir.constant(32 : index) : i64
    // CHECK:  %[[LI:.*]] = llvm.mul %[[INX]], %[[LDM]]  : i64
    // CHECK:  %[[LIJ:.*]] = llvm.add %[[LI]], %[[INX]]  : i64
    // CHECK:  %[[ADDRESS:.*]] = llvm.getelementptr %[[BASE]][%[[LIJ]]] : (!llvm.ptr<f16, 3>, i64) -> !llvm.ptr<f16, 3>
    // CHECK:  %[[LDM32:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:  %[[FRAG:.*]] = nvvm.wmma.load %[[ADDRESS]], %[[LDM32]]
    // CHECK-SAME: {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}  : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
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
    // CHECK32-SAME: {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}  : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK32:  llvm.return %[[FRAG]] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
  }
}

// -----

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_store_op
  // CHECK-SAME: (%[[D:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>)
  // CHECK32-LABEL: func @gpu_wmma_store_op
  // CHECK32-SAME: (%[[D:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>)
  func.func @gpu_wmma_store_op(%arg0 : !gpu.mma_matrix<16x16xf16, "COp">) -> () {
    %sg = memref.alloca(){alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %sg[%i,%j] {leadDimension= 32 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
    // CHECK:  %[[INX:.*]] = llvm.mlir.constant(16 : index) : i64
    // CHECK: %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[{{.*}}, {{.*}}]
    // CHECK:  %[[EL1:.*]] = llvm.extractvalue %[[D]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[EL2:.*]] = llvm.extractvalue %[[D]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[EL3:.*]] = llvm.extractvalue %[[D]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[EL4:.*]] = llvm.extractvalue %[[D]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  %[[BASE:.*]] = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK:  %[[LDM:.*]] = llvm.mlir.constant(32 : index) : i64
    // CHECK:  %[[LI:.*]] = llvm.mul %[[INX]], %[[LDM]]   : i64
    // CHECK:  %[[LIJ:.*]] = llvm.add %[[LI]], %[[INX]]  : i64
    // CHECK:  %[[ADDRESS:.*]] = llvm.getelementptr %[[BASE]][%[[LIJ]]] : (!llvm.ptr<f16, 3>, i64) -> !llvm.ptr<f16, 3>
    // CHECK:  %[[LDM32:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:  nvvm.wmma.store %[[ADDRESS]], %[[LDM32]], %[[EL1]], %[[EL2]], %[[EL3]], %[[EL4]]
    // CHECK-SAME: {eltype = #nvvm.mma_type<f16>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32} : !llvm.ptr<f16, 3>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>
    // CHECK:  llvm.return

    // CHECK32:  %[[INX:.*]] = llvm.mlir.constant(16 : index) : i32
    // CHECK32: %{{.*}} = llvm.insertvalue %{{.*}}, %{{.*}}[{{.*}}, {{.*}}]
    // CHECK32:  %[[EL1:.*]] = llvm.extractvalue %[[D]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK32:  %[[EL2:.*]] = llvm.extractvalue %[[D]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK32:  %[[EL3:.*]] = llvm.extractvalue %[[D]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK32:  %[[EL4:.*]] = llvm.extractvalue %[[D]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK32:  %[[BASE:.*]] = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f16, 3>, ptr<f16, 3>, i32, array<2 x i32>, array<2 x i32>)>
    // CHECK32:  %[[LDM:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK32:  %[[LI:.*]] = llvm.mul %[[INX]], %[[LDM]]   : i32
    // CHECK32:  %[[LIJ:.*]] = llvm.add %[[LI]], %[[INX]]  : i32
    // CHECK32:  %[[ADDRESS:.*]] = llvm.getelementptr %[[BASE]][%[[LIJ]]] : (!llvm.ptr<f16, 3>, i32) -> !llvm.ptr<f16, 3>
    // CHECK32:  %[[LDM32:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK32:  nvvm.wmma.store %[[ADDRESS]], %[[LDM32]], %[[EL1]], %[[EL2]], %[[EL3]], %[[EL4]]
    // CHECK32-SAME: {eltype = #nvvm.mma_type<f16>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32} : !llvm.ptr<f16, 3>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>
    // CHECK32:  llvm.return
    return
  }
}

// -----

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_mma_op
  // CHECK-SAME: (%[[A:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>, %[[B:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>, %[[C:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>)
  func.func @gpu_wmma_mma_op(%A : !gpu.mma_matrix<16x16xf16, "AOp">, %B : !gpu.mma_matrix<16x16xf16, "BOp">, %C : !gpu.mma_matrix<16x16xf16, "COp">) -> (!gpu.mma_matrix<16x16xf16, "COp">) {
    %D = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
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
    // CHECK:  %[[RES:.*]] = nvvm.wmma.mma %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[A8]], %[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[B8]], %[[C1]], %[[C2]], %[[C3]], %[[C4]]
    // CHECK-SAME: {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f16>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32} : (
    // CHECK-SAME: vector<2xf16>, {{.*}}) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  llvm.return %[[RES]] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    return %D : !gpu.mma_matrix<16x16xf16, "COp">
  }
}

// -----

gpu.module @test_module {

// CHECK-LABEL: func @gpu_wmma_mma_loop_op
//       CHECK:   %[[C:.+]] = nvvm.wmma.load %{{.*}}, %{{.*}} {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<c>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32} : (!llvm.ptr<f16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   llvm.br ^bb1(%{{.*}}, %[[C]] : i64, !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>)
//       CHECK:  ^bb1(%{{.*}}: i64, %[[ACC:.+]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>):  // 2 preds: ^bb0, ^bb2
//       CHECK:   llvm.cond_br %{{.*}}, ^bb2, ^bb3
//       CHECK:  ^bb2:  // pred: ^bb1
//       CHECK:   %[[A:.+]] = nvvm.wmma.load %{{.*}}, %{{.*}} {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32} : (!llvm.ptr<f16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B:.+]] = nvvm.wmma.load %{{.*}}, %{{.*}} {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<b>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32} : (!llvm.ptr<f16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[A0:.+]] = llvm.extractvalue %[[A]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[A1:.+]] = llvm.extractvalue %[[A]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[A2:.+]] = llvm.extractvalue %[[A]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[A3:.+]] = llvm.extractvalue %[[A]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[A4:.+]] = llvm.extractvalue %[[A]][4 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[A5:.+]] = llvm.extractvalue %[[A]][5 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[A6:.+]] = llvm.extractvalue %[[A]][6 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[A7:.+]] = llvm.extractvalue %[[A]][7 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B0:.+]] = llvm.extractvalue %[[B]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B1:.+]] = llvm.extractvalue %[[B]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B2:.+]] = llvm.extractvalue %[[B]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B3:.+]] = llvm.extractvalue %[[B]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B4:.+]] = llvm.extractvalue %[[B]][4 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B5:.+]] = llvm.extractvalue %[[B]][5 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B6:.+]] = llvm.extractvalue %[[B]][6 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B7:.+]] = llvm.extractvalue %[[B]][7 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[ACC0:.+]] = llvm.extractvalue %[[ACC]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[ACC1:.+]] = llvm.extractvalue %[[ACC]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[ACC2:.+]] = llvm.extractvalue %[[ACC]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[ACC3:.+]] = llvm.extractvalue %[[ACC]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[ACC_MUL:.+]] = nvvm.wmma.mma %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[B0]], %[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[ACC0]], %[[ACC1]], %[[ACC2]], %[[ACC3]] {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f16>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32} : (vector<2xf16>, {{.*}} -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   llvm.br ^bb1(%{{.*}}, %[[ACC_MUL]] : i64, !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>)
//       CHECK:  ^bb3:  // pred: ^bb1
//       CHECK:   %[[E0:.+]] = llvm.extractvalue %[[ACC]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[E1:.+]] = llvm.extractvalue %[[ACC]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[E2:.+]] = llvm.extractvalue %[[ACC]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[E3:.+]] = llvm.extractvalue %[[ACC]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   nvvm.wmma.store %{{.*}}, %{{.*}}, %[[E0]], %[[E1]], %[[E2]], %[[E3]] {eltype = #nvvm.mma_type<f16>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32} : !llvm.ptr<f16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>

  func.func @gpu_wmma_mma_loop_op(%arg0: memref<128x128xf16>, %arg1: memref<128x128xf16>, %arg2: memref<128x128xf16>) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %0 = gpu.subgroup_mma_load_matrix %arg2[%c0, %c0] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
      cf.br ^bb1(%c0, %0 : index, !gpu.mma_matrix<16x16xf16, "COp">)
    ^bb1(%1: index, %2: !gpu.mma_matrix<16x16xf16, "COp">):  // 2 preds: ^bb0, ^bb2
      %3 = arith.cmpi slt, %1, %c128 : index
      cf.cond_br %3, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %4 = gpu.subgroup_mma_load_matrix %arg0[%c0, %1] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
      %5 = gpu.subgroup_mma_load_matrix %arg1[%1, %c0] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
      %6 = gpu.subgroup_mma_compute %4, %5, %2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
      %7 = arith.addi %1, %c32 : index
      cf.br ^bb1(%7, %6 : index, !gpu.mma_matrix<16x16xf16, "COp">)
    ^bb3:  // pred: ^bb1
      gpu.subgroup_mma_store_matrix %2, %arg2[%c0, %c0] {leadDimension = 128 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<128x128xf16>
      return
    }
}


// -----

gpu.module @test_module {

// CHECK-LABEL: func @gpu_wmma_constant_op
//       CHECK: %[[CST:.+]] = llvm.mlir.constant(1.000000e+00 : f16) : f16
//       CHECK: %[[V0:.+]] = llvm.mlir.undef : vector<2xf16>
//       CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
//       CHECK: %[[V1:.+]] = llvm.insertelement %[[CST]], %[[V0]][%[[C0]] : i32] : vector<2xf16>
//       CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
//       CHECK: %[[V2:.+]] = llvm.insertelement %[[CST]], %[[V1]][%[[C1]] : i32] : vector<2xf16>
//       CHECK: %[[M0:.+]] = llvm.mlir.undef : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[M1:.+]] = llvm.insertvalue %[[V2]], %[[M0]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[M2:.+]] = llvm.insertvalue %[[V2]], %[[M1]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[M3:.+]] = llvm.insertvalue %[[V2]], %[[M2]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[M4:.+]] = llvm.insertvalue %[[V2]], %[[M3]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: llvm.return %[[M4]] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  func.func @gpu_wmma_constant_op()  ->(!gpu.mma_matrix<16x16xf16, "COp">) {
    %cst = arith.constant 1.0 : f16
    %C = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf16, "COp">
    return %C : !gpu.mma_matrix<16x16xf16, "COp">
  }
}

// -----

gpu.module @test_module {

// CHECK-LABEL: func @gpu_wmma_elementwise
//       CHECK: %[[M0:.*]] = llvm.mlir.undef : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[A0:.*]] = llvm.extractvalue %{{.*}}[0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[B0:.*]] = llvm.extractvalue %{{.*}}[0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[C0:.*]] = llvm.fadd %[[A0]], %[[B0]]  : vector<2xf16>
//       CHECK: %[[M1:.*]] = llvm.insertvalue %[[C0]], %[[M0]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[A1:.*]] = llvm.extractvalue %{{.*}}[1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[B1:.*]] = llvm.extractvalue %{{.*}}[1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[C1:.*]] = llvm.fadd %[[A1]], %[[B1]]  : vector<2xf16>
//       CHECK: %[[M2:.*]] = llvm.insertvalue %[[C1]], %[[M1]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[A2:.*]] = llvm.extractvalue %{{.*}}[2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[B2:.*]] = llvm.extractvalue %{{.*}}[2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[C2:.*]] = llvm.fadd %[[A2]], %[[B2]]  : vector<2xf16>
//       CHECK: %[[M3:.*]] = llvm.insertvalue %[[C2]], %[[M2]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[A3:.*]] = llvm.extractvalue %{{.*}}[3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[B3:.*]] = llvm.extractvalue %{{.*}}[3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[C3:.*]] = llvm.fadd %[[A3]], %[[B3]]  : vector<2xf16>
//       CHECK: %[[M4:.*]] = llvm.insertvalue %[[C3]], %[[M3]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>

//       CHECK: %[[M0:.*]] = llvm.mlir.undef : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[A0:.*]] = llvm.extractvalue %{{.*}}[0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[B0:.*]] = llvm.extractvalue %{{.*}}[0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[CMP0:.*]] = llvm.fcmp "ogt" %[[A0]], %[[B0]] : vector<2xf16>
//       CHECK: %[[SEL0:.*]] = llvm.select %[[CMP0]], %[[A0]], %[[B0]] : vector<2xi1>, vector<2xf16>
//       CHECK: %[[CMP1:.*]] = llvm.fcmp "uno" %[[A0]], %[[B0]] : vector<2xf16>
//       CHECK: %[[NAN:.*]] = llvm.mlir.constant(0x7E00 : f16) : vector<2xf16>
//       CHECK: %[[C0:.*]] = llvm.select %[[CMP1]], %[[NAN]], %[[SEL0]] : vector<2xi1>, vector<2xf16>
//       CHECK: %[[M1:.*]] = llvm.insertvalue %[[C0]], %[[M0]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[A1:.*]] = llvm.extractvalue %{{.*}}[1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[B1:.*]] = llvm.extractvalue %{{.*}}[1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[CMP2:.*]] = llvm.fcmp "ogt" %[[A1]], %[[B1]] : vector<2xf16>
//       CHECK: %[[SEL1:.*]] = llvm.select %[[CMP2]], %[[A1]], %[[B1]] : vector<2xi1>, vector<2xf16>
//       CHECK: %[[CMP3:.*]] = llvm.fcmp "uno" %[[A1]], %[[B1]] : vector<2xf16>
//       CHECK: %[[NAN:.*]] = llvm.mlir.constant(0x7E00 : f16) : vector<2xf16>
//       CHECK: %[[C1:.*]] = llvm.select %[[CMP3]], %[[NAN]], %[[SEL1]] : vector<2xi1>, vector<2xf16>
//       CHECK: %[[M2:.*]] = llvm.insertvalue %[[C1]], %[[M1]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[A2:.*]] = llvm.extractvalue %{{.*}}[2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[B2:.*]] = llvm.extractvalue %{{.*}}[2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[CMP4:.*]] = llvm.fcmp "ogt" %[[A2]], %[[B2]] : vector<2xf16>
//       CHECK: %[[SEL2:.*]] = llvm.select %[[CMP4]], %[[A2]], %[[B2]] : vector<2xi1>, vector<2xf16>
//       CHECK: %[[CMP5:.*]] = llvm.fcmp "uno" %[[A2]], %[[B2]] : vector<2xf16>
//       CHECK: %[[NAN:.*]] = llvm.mlir.constant(0x7E00 : f16) : vector<2xf16>
//       CHECK: %[[C2:.*]] = llvm.select %[[CMP5]], %[[NAN]], %[[SEL2]] : vector<2xi1>, vector<2xf16>
//       CHECK: %[[M3:.*]] = llvm.insertvalue %[[C2]], %[[M2]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[A3:.*]] = llvm.extractvalue %{{.*}}[3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[B3:.*]] = llvm.extractvalue %{{.*}}[3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK: %[[CMP6:.*]] = llvm.fcmp "ogt" %[[A3]], %[[B3]] : vector<2xf16>
//       CHECK: %[[SEL3:.*]] = llvm.select %[[CMP6]], %[[A3]], %[[B3]] : vector<2xi1>, vector<2xf16>
//       CHECK: %[[CMP7:.*]] = llvm.fcmp "uno" %[[A3]], %[[B3]] : vector<2xf16>
//       CHECK: %[[NAN:.*]] = llvm.mlir.constant(0x7E00 : f16) : vector<2xf16>
//       CHECK: %[[C3:.*]] = llvm.select %[[CMP7]], %[[NAN]], %[[SEL3]] : vector<2xi1>, vector<2xf16>
//       CHECK: %[[M5:.*]] = llvm.insertvalue %[[C3]], %[[M3]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>

//       CHECK: llvm.return %[[M5]] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  func.func @gpu_wmma_elementwise(%A : !gpu.mma_matrix<16x16xf16, "COp">, %B : !gpu.mma_matrix<16x16xf16, "COp">)  ->(!gpu.mma_matrix<16x16xf16, "COp">) {
    %C = gpu.subgroup_mma_elementwise addf %A, %B : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
    %D = gpu.subgroup_mma_elementwise maxf %C, %B : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
    return %D : !gpu.mma_matrix<16x16xf16, "COp">
  }
}
