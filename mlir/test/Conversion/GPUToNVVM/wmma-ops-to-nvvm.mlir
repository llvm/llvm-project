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
  func @gpu_wmma_store_op(%arg0 : !gpu.mma_matrix<16x16xf16, "COp">) -> () {
    %sg = memref.alloca(){alignment = 32} : memref<32x32xf16, 3>
    %i = constant 16 : index
    %j = constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %sg[%i,%j] {leadDimension= 32 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
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
  // CHECK-SAME: (%[[A:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>, %[[B:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>, %[[C:.*]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>)
  func @gpu_wmma_mma_op(%A : !gpu.mma_matrix<16x16xf16, "AOp">, %B : !gpu.mma_matrix<16x16xf16, "BOp">, %C : !gpu.mma_matrix<16x16xf16, "COp">) -> (!gpu.mma_matrix<16x16xf16, "COp">) {
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
    // CHECK:  %[[RES:.*]] = nvvm.wmma.m16n16k16.mma.row.row.f16.f16 %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[A8]], %[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[B8]], %[[C1]], %[[C2]], %[[C3]], %[[C4]] : vector<2xf16> -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    // CHECK:  llvm.return %[[RES]] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    return %D : !gpu.mma_matrix<16x16xf16, "COp">
  }
}

// -----

gpu.module @test_module {

// CHECK-LABEL: func @gpu_wmma_mma_loop_op
//       CHECK:   %[[C:.+]] = nvvm.wmma.m16n16k16.load.c.f16.row.stride %{{.*}}, %{{.*}} : (!llvm.ptr<i32>, i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   llvm.br ^bb1(%{{.*}}, %[[C]] : i32, !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>)
//       CHECK:  ^bb1(%{{.*}}: i32, %[[ACC:.+]]: !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>):  // 2 preds: ^bb0, ^bb2
//       CHECK:   llvm.cond_br %38, ^bb2, ^bb3
//       CHECK:  ^bb2:  // pred: ^bb1
//       CHECK:   %[[A:.+]] = nvvm.wmma.m16n16k16.load.a.f16.row.stride %{{.*}}, %{{.*}} : (!llvm.ptr<i32>, i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %[[B:.+]] = nvvm.wmma.m16n16k16.load.b.f16.row.stride %{{.*}}, %{{.*}} : (!llvm.ptr<i32>, i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
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
//       CHECK:   %[[ACC_MUL:.+]] = nvvm.wmma.m16n16k16.mma.row.row.f16.f16 %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[B0]], %[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[ACC0]], %[[ACC1]], %[[ACC2]], %[[ACC3]] : vector<2xf16> -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   llvm.br ^bb1(%{{.*}}, %[[ACC_MUL]] : i32, !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>)
//       CHECK:  ^bb3:  // pred: ^bb1
//       CHECK:   %87 = llvm.extractvalue %[[ACC]][0 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %88 = llvm.extractvalue %[[ACC]][1 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %89 = llvm.extractvalue %[[ACC]][2 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   %90 = llvm.extractvalue %[[ACC]][3 : i32] : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
//       CHECK:   nvvm.wmma.m16n16k16.store.d.f16.row.stride %86, %87, %88, %89, %90, %79 : !llvm.ptr<i32>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, i32

  func @gpu_wmma_mma_loop_op(%arg0: memref<128x128xf16>, %arg1: memref<128x128xf16>, %arg2: memref<128x128xf16>) {
      %c0 = constant 0 : index
      %c128 = constant 128 : index
      %c32 = constant 32 : index
      %0 = gpu.subgroup_mma_load_matrix %arg2[%c0, %c0] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
      br ^bb1(%c0, %0 : index, !gpu.mma_matrix<16x16xf16, "COp">)
    ^bb1(%1: index, %2: !gpu.mma_matrix<16x16xf16, "COp">):  // 2 preds: ^bb0, ^bb2
      %3 = cmpi slt, %1, %c128 : index
      cond_br %3, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %4 = gpu.subgroup_mma_load_matrix %arg0[%c0, %1] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
      %5 = gpu.subgroup_mma_load_matrix %arg1[%1, %c0] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
      %6 = gpu.subgroup_mma_compute %4, %5, %2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
      %7 = addi %1, %c32 : index
      br ^bb1(%7, %6 : index, !gpu.mma_matrix<16x16xf16, "COp">)
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
  func @gpu_wmma_constant_op()  ->(!gpu.mma_matrix<16x16xf16, "COp">) {
    %cst = constant 1.0 : f16
    %C = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf16, "COp">
    return %C : !gpu.mma_matrix<16x16xf16, "COp">
  }
}
