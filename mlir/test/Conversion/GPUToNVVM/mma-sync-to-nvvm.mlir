// RUN: mlir-opt --convert-gpu-to-nvvm --split-input-file %s | FileCheck %s

gpu.module @test_module {
  // CHECK-LABEL: @m16n8k16_fp16
  func @m16n8k16_fp16(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
    // CHECK: llvm.extractvalue %arg0[0] : !llvm.array<4 x vector<2xf16>>
    // CHECK: llvm.extractvalue %arg0[1] : !llvm.array<4 x vector<2xf16>>
    // CHECK: llvm.extractvalue %arg0[2] : !llvm.array<4 x vector<2xf16>>
    // CHECK: llvm.extractvalue %arg0[3] : !llvm.array<4 x vector<2xf16>>

    // CHECK: llvm.extractvalue %arg1[0] : !llvm.array<2 x vector<2xf16>>
    // CHECK: llvm.extractvalue %arg1[1] : !llvm.array<2 x vector<2xf16>>

    // CHECK: llvm.extractvalue %arg2[0] : !llvm.array<2 x vector<2xf16>>
    // CHECK: llvm.extractvalue %arg2[1] : !llvm.array<2 x vector<2xf16>>
    // CHECK-NOT llvm.extractvalue
    // CHECK: [[d:%.+]] = nvvm.mma.sync
    // CHECK-SAME: shape = {k = 16 : i32, m = 16 : i32, n  = 8 : i32}
    %d = gpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>    
    // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>    
    // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    // CHECK: llvm.mlir.undef : !llvm.array<2 x vector<2xf16>>
    // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<2 x vector<2xf16>>
    // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[1] : !llvm.array<2 x vector<2xf16>>      
    // CHECK: llvm.return {{%.+}} : !llvm.array<2 x vector<2xf16>>
    return %d : vector<2x2xf16>
  }

  // CHECK-LABEL: @m16n8k8_fp16
  func @m16n8k8_fp16(%arg0: vector<2x2xf16>, %arg1: vector<1x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
    // CHECK: llvm.extractvalue %arg0[0] : !llvm.array<2 x vector<2xf16>>
    // CHECK: llvm.extractvalue %arg0[1] : !llvm.array<2 x vector<2xf16>>

    // CHECK: llvm.extractvalue %arg1[0] : !llvm.array<1 x vector<2xf16>>

    // CHECK: llvm.extractvalue %arg2[0] : !llvm.array<2 x vector<2xf16>>
    // CHECK: llvm.extractvalue %arg2[1] : !llvm.array<2 x vector<2xf16>>
    // CHECK-NOT llvm.extractvalue
    // CHECK: [[d:%.+]] = nvvm.mma.sync
    // CHECK-SAME: shape = {k = 8 : i32, m = 16 : i32, n  = 8 : i32}
    %d = gpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 8]} : (vector<2x2xf16>, vector<1x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>    
    // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>    
    // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    // CHECK: llvm.mlir.undef : !llvm.array<2 x vector<2xf16>>
    // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<2 x vector<2xf16>>
    // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[1] : !llvm.array<2 x vector<2xf16>>      
    // CHECK: llvm.return {{%.+}} : !llvm.array<2 x vector<2xf16>>
    return %d : vector<2x2xf16>
  }

  // CHECK-LABEL: @m16n8k32_int8
  func @m16n8k32_int8(%arg0: vector<4x4xi8>, %arg1: vector<2x4xi8>, %arg2: vector<2x2xi32>) -> vector<2x2xi32> {

    // CHECK: [[el:%.+]] = llvm.extractvalue %arg0[{{.*}}] : !llvm.array<4 x vector<4xi8>>
    // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
    // CHECK: [[el:%.+]] = llvm.extractvalue %arg0[{{.*}}] : !llvm.array<4 x vector<4xi8>>
    // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
    // CHECK: [[el:%.+]] = llvm.extractvalue %arg0[{{.*}}] : !llvm.array<4 x vector<4xi8>>
    // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
    // CHECK: [[el:%.+]] = llvm.extractvalue %arg0[{{.*}}] : !llvm.array<4 x vector<4xi8>>
    // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32

    // CHECK: [[el:%.+]] = llvm.extractvalue %arg1[{{.*}}] : !llvm.array<2 x vector<4xi8>>
    // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
    // CHECK: [[el:%.+]] = llvm.extractvalue %arg1[{{.*}}] : !llvm.array<2 x vector<4xi8>>
    // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32

    // CHECK: [[el:%.+]] = llvm.extractvalue %arg2[{{.*}}] : !llvm.array<2 x vector<2xi32>>
    // CHECK: [[el:%.+]] = llvm.extractvalue %arg2[{{.*}}] : !llvm.array<2 x vector<2xi32>>

    // CHECK: [[d:%.+]] = nvvm.mma.sync
    // CHECK-SAME: intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>
    // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<s8>
    // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<s8>
    // CHECK-SAME: shape = {k = 32 : i32, m = 16 : i32, n = 8 : i32}
    %d = gpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<4x4xi8>, vector<2x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>

    // CHECK: llvm.return {{%.+}} : !llvm.array<2 x vector<2xi32>>
    return %d : vector<2x2xi32>
  }

  // CHECK-LABEL: @m8n8k4_f64
  func @m8n8k4_f64(%arg0: vector<1x1xf64>, %arg1: vector<1x1xf64>, %arg2: vector<1x2xf64>) -> vector<1x2xf64> {
    // CHECK: llvm.extractvalue %arg0
    // CHECK: llvm.extractvalue %arg1
    // CHECK: llvm.extractvalue %arg2

    // CHECK: [[d:%.+]] = nvvm.mma.sync A[{{%.+}}] B[{{%.+}}] C[{{%.+}}, {{%.+}}]
    // CHECK-SAME: shape = {k = 4 : i32, m = 8 : i32, n = 8 : i32}
    %d = gpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [8, 8, 4]} : (vector<1x1xf64>, vector<1x1xf64>, vector<1x2xf64>) -> vector<1x2xf64>    
    // CHECK: llvm.mlir.undef : vector<2xf64>
    // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(f64, f64)>    
    // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(f64, f64)>
    // CHECK-COUNT-2: llvm.insertelement {{.*}} : vector<2xf64>
    // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<1 x vector<2xf64>>
    // CHECK: llvm.return {{%.+}} : !llvm.array<1 x vector<2xf64>>
    return %d : vector<1x2xf64>
  }

  // CHECK-LABEL: @ldmatrix_x4
  func @ldmatrix_x4(%arg0: memref<128x128xf16, 3>) ->  vector<4x2xf16> {
    %c0  = arith.constant 0 : index
    // CHECK: nvvm.ldmatrix {{%.+}} {layout = #nvvm.mma_layout<row>, num = 4 : i32} {{.*}} -> !llvm.struct<(i32, i32, i32, i32)
    %a = gpu.mma.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf16, 3> -> vector<4x2xf16>
    // CHECK: llvm.extractvalue
    // CHECK: llvm.bitcast
    // CHECK: llvm.insertvalue
    // CHECK: llvm.extractvalue
    // CHECK: llvm.bitcast
    // CHECK: llvm.insertvalue
    // CHECK: llvm.extractvalue
    // CHECK: llvm.bitcast
    // CHECK: llvm.insertvalue
    // CHECK: llvm.extractvalue
    // CHECK: llvm.bitcast
    // CHECK: llvm.insertvalue
    return %a : vector<4x2xf16>
  }

  // CHECK-LABEL: @ldmatrix_x1
  func @ldmatrix_x1(%arg0: memref<128x128xf16, 3>) ->  vector<1x2xf16> {
    %c0  = arith.constant 0 : index
    // CHECK: nvvm.ldmatrix {{%.+}} {layout = #nvvm.mma_layout<row>, num = 1 : i32} {{.*}} -> i32
    %a = gpu.mma.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 1 : i32} : memref<128x128xf16, 3> -> vector<1x2xf16>    
    // CHECK: llvm.bitcast
    // CHECK: llvm.insertvalue    
    return %a : vector<1x2xf16>
  }
}
