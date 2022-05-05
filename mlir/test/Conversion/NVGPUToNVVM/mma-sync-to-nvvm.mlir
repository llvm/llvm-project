// RUN: mlir-opt --convert-nvgpu-to-nvvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: @m16n8k16_fp16
func.func @m16n8k16_fp16(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[2] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[3] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK-NOT llvm.extractvalue
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: shape = {k = 16 : i32, m = 16 : i32, n  = 8 : i32}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>    
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>    
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  // CHECK: llvm.mlir.undef : !llvm.array<2 x vector<2xf16>>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[1] : !llvm.array<2 x vector<2xf16>>      
  return %d : vector<2x2xf16>
}

// -----

// Same as above but with fp32 acumulation type.

// CHECK-LABEL: @m16n8k16_fp16_fp32
func.func @m16n8k16_fp16_fp32(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // We just need to check the mma instruction and the manipulatin of the result.
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: shape = {k = 16 : i32, m = 16 : i32, n  = 8 : i32}
  // CHECK-SAME: (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>    
  // CHECK: [[undef:%.+]] = llvm.mlir.undef : vector<2xf32>
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK: [[d00:%.+]] = llvm.insertelement {{%.+}}, [[undef]][{{.*}}] : vector<2xf32>
  // CHECK: [[d01:%.+]] = llvm.insertelement {{%.+}}, [[d00]][{{.*}}] : vector<2xf32>

  // CHECK: [[undef:%.+]] = llvm.mlir.undef : vector<2xf32>  
  // CHECK-DAG: llvm.extractvalue [[d]][2] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK-DAG: llvm.extractvalue [[d]][3] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK: [[d10:%.+]] = llvm.insertelement {{%.+}}, [[undef]][{{.*}}] : vector<2xf32>
  // CHECK: [[d11:%.+]] = llvm.insertelement {{%.+}}, [[d10]][{{.*}}] : vector<2xf32>
  
  // CHECK-DAG: llvm.insertvalue [[d01]], {{%.+}}[0] : !llvm.array<2 x vector<2xf32>>
  // CHECK-DAG: llvm.insertvalue [[d11]], {{%.+}}[1] : !llvm.array<2 x vector<2xf32>>      
  return %d : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @m16n8k8_fp16
func.func @m16n8k8_fp16(%arg0: vector<2x2xf16>, %arg1: vector<1x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK-NOT llvm.extractvalue
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: shape = {k = 8 : i32, m = 16 : i32, n  = 8 : i32}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 8]} : (vector<2x2xf16>, vector<1x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>    
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>    
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  // CHECK: llvm.mlir.undef : !llvm.array<2 x vector<2xf16>>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[1] : !llvm.array<2 x vector<2xf16>>      
  // CHECK: return
  return %d : vector<2x2xf16>
}

// -----


// CHECK-LABEL: @m16n8k32_int8
func.func @m16n8k32_int8(%arg0: vector<4x4xi8>, %arg1: vector<2x4xi8>, %arg2: vector<2x2xi32>) -> vector<2x2xi32> {
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<s8>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<s8>
  // CHECK-SAME: shape = {k = 32 : i32, m = 16 : i32, n = 8 : i32}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<4x4xi8>, vector<2x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  return %d : vector<2x2xi32>
}

// -----

// CHECK-LABEL: @m8n8k4_f64
func.func @m8n8k4_f64(%arg0: vector<1x1xf64>, %arg1: vector<1x1xf64>, %arg2: vector<1x2xf64>) -> vector<1x2xf64> {
  // CHECK: llvm.extractvalue
  // CHECK: llvm.extractvalue
  // CHECK: llvm.extractvalue
  // CHECK: [[d:%.+]] = nvvm.mma.sync A[{{%.+}}] B[{{%.+}}] C[{{%.+}}, {{%.+}}]
  // CHECK-SAME: shape = {k = 4 : i32, m = 8 : i32, n = 8 : i32}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [8, 8, 4]} : (vector<1x1xf64>, vector<1x1xf64>, vector<1x2xf64>) -> vector<1x2xf64>
  // CHECK: llvm.mlir.undef : vector<2xf64>
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(f64, f64)>    
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(f64, f64)>
  // CHECK-COUNT-2: llvm.insertelement {{.*}} : vector<2xf64>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<1 x vector<2xf64>>
  // CHECK: return
  return %d : vector<1x2xf64>
}

// -----


// CHECK-LABEL: @ldmatrix_x4
func.func @ldmatrix_x4(%arg0: memref<128x128xf16, 3>) ->  vector<4x2xf16> {
  %c0  = arith.constant 0 : index
  // CHECK: nvvm.ldmatrix {{%.+}} {layout = #nvvm.mma_layout<row>, num = 4 : i32} {{.*}} -> !llvm.struct<(i32, i32, i32, i32)
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf16, 3> -> vector<4x2xf16>
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

// -----

// CHECK-LABEL: @ldmatrix_x1
func.func @ldmatrix_x1(%arg0: memref<128x128xf16, 3>) ->  vector<1x2xf16> {
  %c0  = arith.constant 0 : index
  // CHECK: nvvm.ldmatrix {{%.+}} {layout = #nvvm.mma_layout<row>, num = 1 : i32} {{.*}} -> i32
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 1 : i32} : memref<128x128xf16, 3> -> vector<1x2xf16>    
  // CHECK: llvm.bitcast
  // CHECK: llvm.insertvalue    
  return %a : vector<1x2xf16>
}

// -----

// CHECK-LABEL: @m16n8k4_tf32
func.func @m16n8k4_tf32(%arg0: vector<2x1xf32>, %arg1: vector<1x1xf32>, %arg2: vector<4x1xf32>) -> vector<4x1xf32> {  
  // The A, B operand should be bitcast to i32
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32  
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32

  // CHECK: [[d:%.+]] = nvvm.mma.sync A[{{%.+}}, {{%.+}}] B[{{%.+}}] C[{{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}]
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<tf32>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<tf32>
  // CHECK-SAME: shape = {k = 4 : i32, m = 16 : i32, n = 8 : i32}
  // CHECK-SAME: -> !llvm.struct<(f32, f32, f32, f32)>  
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 4]} : (vector<2x1xf32>, vector<1x1xf32>, vector<4x1xf32>) -> vector<4x1xf32>  
  // CHECK: [[el:%.+]] = llvm.extractvalue [[d]][0]
  // CHECK: llvm.bitcast [[el]] : f32 to vector<1xf32>
  // CHECK: [[el:%.+]] = llvm.extractvalue [[d]][1]
  // CHECK: llvm.bitcast [[el]] : f32 to vector<1xf32>
  // CHECK: [[el:%.+]] = llvm.extractvalue [[d]][2]
  // CHECK: llvm.bitcast [[el]] : f32 to vector<1xf32>
  // CHECK: [[el:%.+]] = llvm.extractvalue [[d]][3]
  // CHECK: llvm.bitcast [[el]] : f32 to vector<1xf32>
  // CHECK-COUNT-4: llvm.insertvalue {{.*}} : !llvm.array<4 x vector<1xf32>>
  return %d : vector<4x1xf32>
}