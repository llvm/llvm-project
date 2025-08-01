// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_f32
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[ZERO]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_f32(%v: vector<2xf32>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf32> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_f32_vec1
// CHECK-DAG:   [[ZERO_I16:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf32>
// CHECK:       [[ZERO_F32:%.+]] = llvm.mlir.zero : vector<2xf32>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F32]]{{\[}}[[C0_I32]] : i32] : vector<2xf32>
// CHECK-DAG:   [[C0_I32_2:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C0_I32_2]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C1_I32]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[ZERO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_f32_vec1(%v: vector<1xf32>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xf32> to vector<4xf8E4M3FN>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf32> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_f32_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<4xi8> to vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf32>
// CHECK:       [[ZERO_F32:%.+]] = llvm.mlir.zero : vector<2xf32>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F32]]{{\[}}[[C0_I32]] : i32] : vector<2xf32>
// CHECK-DAG:   [[C0_I32_2:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C0_I32_2]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C1_I32]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f32 [[ELEM0]], [[ELEM1]], %arg2 -> [[EXISTING_BITCAST]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_into_f8e4m3_f32_vec1(%v: vector<1xf32>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xf32> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_f16
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f16 %arg0, %arg1 -> [[ZERO]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_f16(%v: vector<2xf16>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf16> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_f16_vec1
// CHECK-DAG:   [[ZERO_I16:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf16>
// CHECK:       [[ZERO_F16:%.+]] = llvm.mlir.zero : vector<2xf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F16]]{{\[}}[[C0_I32]] : i32] : vector<2xf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f16 [[INSERT]], %arg1 -> [[ZERO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_f16_vec1(%v: vector<1xf16>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xf16> to vector<4xf8E4M3FN>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf16> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_f16_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<4xi8> to vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf16>
// CHECK:       [[ZERO_F16:%.+]] = llvm.mlir.zero : vector<2xf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F16]]{{\[}}[[C0_I32]] : i32] : vector<2xf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.f16 [[INSERT]], %arg2 -> [[EXISTING_BITCAST]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_into_f8e4m3_f16_vec1(%v: vector<1xf16>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xf16> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_bf16
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 %arg0, %arg1 -> [[ZERO]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xbf16> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_bf16_vec1
// CHECK-DAG:   [[ZERO_I16:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xbf16>
// CHECK:       [[ZERO_BF16:%.+]] = llvm.mlir.zero : vector<2xbf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_BF16]]{{\[}}[[C0_I32]] : i32] : vector<2xbf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 [[INSERT]], %arg1 -> [[ZERO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_f8e4m3_bf16_vec1(%v: vector<1xbf16>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xbf16> to vector<4xf8E4M3FN>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xbf16> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_bf16_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<4xi8> to vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xbf16>
// CHECK:       [[ZERO_BF16:%.+]] = llvm.mlir.zero : vector<2xbf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_BF16]]{{\[}}[[C0_I32]] : i32] : vector<2xbf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 [[INSERT]], %arg2 -> [[EXISTING_BITCAST]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       return [[CAST]] : vector<4xf8E4M3FN>
func.func @packed_scaled_trunc_into_f8e4m3_bf16_vec1(%v: vector<1xbf16>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xbf16> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_f32
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[ZERO]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_f32(%v: vector<2xf32>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf32> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_f32_vec1
// CHECK-DAG:   [[ZERO_I16:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf32>
// CHECK:       [[ZERO_F32:%.+]] = llvm.mlir.zero : vector<2xf32>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F32]]{{\[}}[[C0_I32]] : i32] : vector<2xf32>
// CHECK-DAG:   [[C0_I32_2:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C0_I32_2]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C1_I32]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[ZERO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_f32_vec1(%v: vector<1xf32>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xf32> to vector<4xf8E5M2>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf32> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_f32_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<4xi8> to vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf32>
// CHECK:       [[ZERO_F32:%.+]] = llvm.mlir.zero : vector<2xf32>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F32]]{{\[}}[[C0_I32]] : i32] : vector<2xf32>
// CHECK-DAG:   [[C0_I32_2:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C0_I32_2]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C1_I32]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f32 [[ELEM0]], [[ELEM1]], %arg2 -> [[EXISTING_BITCAST]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_into_f8e5m2_f32_vec1(%v: vector<1xf32>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xf32> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_f16
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f16 %arg0, %arg1 -> [[ZERO]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_f16(%v: vector<2xf16>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf16> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_f16_vec1
// CHECK-DAG:   [[ZERO_I16:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf16>
// CHECK:       [[ZERO_F16:%.+]] = llvm.mlir.zero : vector<2xf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F16]]{{\[}}[[C0_I32]] : i32] : vector<2xf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f16 [[INSERT]], %arg1 -> [[ZERO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_f16_vec1(%v: vector<1xf16>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xf16> to vector<4xf8E5M2>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf16> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_f16_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<4xi8> to vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf16>
// CHECK:       [[ZERO_F16:%.+]] = llvm.mlir.zero : vector<2xf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F16]]{{\[}}[[C0_I32]] : i32] : vector<2xf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.f16 [[INSERT]], %arg2 -> [[EXISTING_BITCAST]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_into_f8e5m2_f16_vec1(%v: vector<1xf16>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xf16> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_bf16
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.bf16 %arg0, %arg1 -> [[ZERO]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xbf16> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_bf16_vec1
// CHECK-DAG:   [[ZERO_I16:%.+]] = llvm.mlir.zero : vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xbf16>
// CHECK:       [[ZERO_BF16:%.+]] = llvm.mlir.zero : vector<2xbf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_BF16]]{{\[}}[[C0_I32]] : i32] : vector<2xbf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.bf16 [[INSERT]], %arg1 -> [[ZERO_I16]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_f8e5m2_bf16_vec1(%v: vector<1xbf16>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xbf16> to vector<4xf8E5M2>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xbf16> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_bf16_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<4xi8> to vector<2xi16>
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xbf16>
// CHECK:       [[ZERO_BF16:%.+]] = llvm.mlir.zero : vector<2xbf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_BF16]]{{\[}}[[C0_I32]] : i32] : vector<2xbf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.bf8.bf16 [[INSERT]], %arg2 -> [[EXISTING_BITCAST]][false] : vector<2xi16>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : vector<2xi16> to vector<4xi8>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<4xi8> to vector<4xf8E5M2>
// CHECK:       return [[CAST]] : vector<4xf8E5M2>
func.func @packed_scaled_trunc_into_f8e5m2_bf16_vec1(%v: vector<1xbf16>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xbf16> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_f32
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : i32
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement %arg0{{\[}}[[C0]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement %arg0{{\[}}[[C1]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[ZERO]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_f32(%v: vector<2xf32>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf32> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_f32_vec1
// CHECK-DAG:   [[ZERO_I32:%.+]] = llvm.mlir.zero : i32
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf32>
// CHECK:       [[ZERO_F32:%.+]] = llvm.mlir.zero : vector<2xf32>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F32]]{{\[}}[[C0_I32]] : i32] : vector<2xf32>
// CHECK-DAG:   [[C0_I32_2:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C0_I32_2]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C1_I32]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f32 [[ELEM0]], [[ELEM1]], %arg1 -> [[ZERO_I32]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_f32_vec1(%v: vector<1xf32>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xf32> to vector<8xf4E2M1FN>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf32> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_f32_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<8xi4> to i32
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf32>
// CHECK:       [[ZERO_F32:%.+]] = llvm.mlir.zero : vector<2xf32>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F32]]{{\[}}[[C0_I32]] : i32] : vector<2xf32>
// CHECK-DAG:   [[C0_I32_2:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM0:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C0_I32_2]] : i32] : vector<2xf32>
// CHECK:       [[ELEM1:%.+]] = llvm.extractelement [[INSERT]]{{\[}}[[C1_I32]] : i32] : vector<2xf32>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f32 [[ELEM0]], [[ELEM1]], %arg2 -> [[EXISTING_BITCAST]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_into_f4e2m1_f32_vec1(%v: vector<1xf32>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xf32> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_f16
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : i32
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f16 %arg0, %arg1 -> [[ZERO]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_f16(%v: vector<2xf16>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf16> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_f16_vec1
// CHECK-DAG:   [[ZERO_I32:%.+]] = llvm.mlir.zero : i32
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf16>
// CHECK:       [[ZERO_F16:%.+]] = llvm.mlir.zero : vector<2xf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F16]]{{\[}}[[C0_I32]] : i32] : vector<2xf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f16 [[INSERT]], %arg1 -> [[ZERO_I32]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_f16_vec1(%v: vector<1xf16>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xf16> to vector<8xf4E2M1FN>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf16> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_f16_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<8xi4> to i32
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xf16>
// CHECK:       [[ZERO_F16:%.+]] = llvm.mlir.zero : vector<2xf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_F16]]{{\[}}[[C0_I32]] : i32] : vector<2xf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.f16 [[INSERT]], %arg2 -> [[EXISTING_BITCAST]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_into_f4e2m1_f16_vec1(%v: vector<1xf16>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xf16> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_bf16
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : i32
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 %arg0, %arg1 -> [[ZERO]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xbf16> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_bf16_vec1
// CHECK-DAG:   [[ZERO_I32:%.+]] = llvm.mlir.zero : i32
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xbf16>
// CHECK:       [[ZERO_BF16:%.+]] = llvm.mlir.zero : vector<2xbf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_BF16]]{{\[}}[[C0_I32]] : i32] : vector<2xbf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 [[INSERT]], %arg1 -> [[ZERO_I32]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_f4e2m1_bf16_vec1(%v: vector<1xbf16>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<1xbf16> to vector<8xf4E2M1FN>
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
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xbf16> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_bf16_vec1
// CHECK-DAG:   [[EXISTING_CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[EXISTING_BITCAST:%.+]] = llvm.bitcast [[EXISTING_CAST]] : vector<8xi4> to i32
// CHECK-DAG:   [[C0_I32:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[EXTRACT:%.+]] = llvm.extractelement %arg0{{\[}}[[C0_I32]] : i32] : vector<1xbf16>
// CHECK:       [[ZERO_BF16:%.+]] = llvm.mlir.zero : vector<2xbf16>
// CHECK:       [[INSERT:%.+]] = llvm.insertelement [[EXTRACT]], [[ZERO_BF16]]{{\[}}[[C0_I32]] : i32] : vector<2xbf16>
// CHECK:       [[CVT:%.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 [[INSERT]], %arg2 -> [[EXISTING_BITCAST]][0] : i32
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[CVT]] : i32 to vector<8xi4>
// CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast [[BITCAST]] : vector<8xi4> to vector<8xf4E2M1FN>
// CHECK:       return [[CAST]] : vector<8xf4E2M1FN>
func.func @packed_scaled_trunc_into_f4e2m1_bf16_vec1(%v: vector<1xbf16>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<1xbf16> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}
