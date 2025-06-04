// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_f32
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : vector<2xi16>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[UNDEF]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_f32(%v: vector<2xf32>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xf32> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_f32
// CHECK-DAG:   [[EXISTING_CAST_TO_I8:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST_TO_I16:%.+]] = llvm.bitcast [[EXISTING_CAST_TO_I8]] : vector<4xi8> to vector<2xi16>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f32 [[ELEM0]], [[ELEM1]], %arg2 -> [[EXISTING_BITCAST_TO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_into_f8e4m3_f32(%v: vector<2xf32>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xf32> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_f16
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f16 %arg0, %arg1 -> [[UNDEF]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_f16(%v: vector<2xf16>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xf16> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_f16
// CHECK-DAG:   [[EXISTING_CAST_TO_I8:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST_TO_I16:%.+]] = llvm.bitcast [[EXISTING_CAST_TO_I8]] : vector<4xi8> to vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f16 %arg0, %arg2 -> [[EXISTING_BITCAST_TO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_into_f8e4m3_f16(%v: vector<2xf16>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xf16> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_bf16
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 %arg0, %arg1 -> [[UNDEF]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xbf16> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_bf16
// CHECK-DAG:   [[EXISTING_CAST_TO_I8:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST_TO_I16:%.+]] = llvm.bitcast [[EXISTING_CAST_TO_I8]] : vector<4xi8> to vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 %arg0, %arg2 -> [[EXISTING_BITCAST_TO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_into_f8e4m3_bf16(%v: vector<2xbf16>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xbf16> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_f32
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : vector<2xi16>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[UNDEF]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_f32(%v: vector<2xf32>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xf32> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_f32
// CHECK-DAG:   [[EXISTING_CAST_TO_I8:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST_TO_I16:%.+]] = llvm.bitcast [[EXISTING_CAST_TO_I8]] : vector<4xi8> to vector<2xi16>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f32 [[ELEM0]], [[ELEM1]], %arg2 -> [[EXISTING_BITCAST_TO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_into_f8e5m2_f32(%v: vector<2xf32>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xf32> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_f16
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f16 %arg0, %arg1 -> [[UNDEF]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_f16(%v: vector<2xf16>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xf16> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_f16
// CHECK-DAG:   [[EXISTING_CAST_TO_I8:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST_TO_I16:%.+]] = llvm.bitcast [[EXISTING_CAST_TO_I8]] : vector<4xi8> to vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f16 %arg0, %arg2 -> [[EXISTING_BITCAST_TO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_into_f8e5m2_f16(%v: vector<2xf16>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xf16> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_bf16
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.bf16 %arg0, %arg1 -> [[UNDEF]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xbf16> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_bf16
// CHECK-DAG:   [[EXISTING_CAST_TO_I8:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST_TO_I16:%.+]] = llvm.bitcast [[EXISTING_CAST_TO_I8]] : vector<4xi8> to vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.bf16 %arg0, %arg2 -> [[EXISTING_BITCAST_TO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_into_f8e5m2_bf16(%v: vector<2xbf16>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xbf16> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_f32
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : i32
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[UNDEF]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_f32(%v: vector<2xf32>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xf32> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_f32
// CHECK-DAG:   [[BITCAST_I4:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[BITCAST_I32:%.+]] = llvm.bitcast [[BITCAST_I4]] : vector<8xi4> to i32
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f32 [[ELEM0]], [[ELEM1]], %arg2 -> [[BITCAST_I32]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_into_f4e2m1_f32(%v: vector<2xf32>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xf32> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_f16
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : i32
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f16 %arg0, %arg1 -> [[UNDEF]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_f16(%v: vector<2xf16>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xf16> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_f16
// CHECK-DAG:   [[BITCAST_I4:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[BITCAST_I32:%.+]] = llvm.bitcast [[BITCAST_I4]] : vector<8xi4> to i32
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f16 %arg0, %arg2 -> [[BITCAST_I32]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_into_f4e2m1_f16(%v: vector<2xf16>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xf16> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_bf16
// CHECK-DAG:   [[UNDEF:%.+]] = llvm.mlir.undef : i32
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 %arg0, %arg1 -> [[UNDEF]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[index 0], %scale : vector<2xbf16> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_bf16
// CHECK-DAG:   [[BITCAST_I4:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[BITCAST_I32:%.+]] = llvm.bitcast [[BITCAST_I4]] : vector<8xi4> to i32
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 %arg0, %arg2 -> [[BITCAST_I32]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_into_f4e2m1_bf16(%v: vector<2xbf16>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[index 0], %scale : vector<2xbf16> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}
