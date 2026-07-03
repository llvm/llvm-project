// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s

// CHECK-LABEL: func.func @scaled_ext_full_f8e4m3_f32
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.fp8 [[BITCAST]][false], %arg1 : vector<2xf32>
func.func @scaled_ext_full_f8e4m3_f32(%v: vector<4xf8E4M3FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e4m3_f16
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.fp8 [[BITCAST]][false], %arg1 : vector<2xf16>
func.func @scaled_ext_full_f8e4m3_f16(%v: vector<4xf8E4M3FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E4M3FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e4m3_bf16
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.fp8 [[BITCAST]][false], %arg1 : vector<2xbf16>
func.func @scaled_ext_full_f8e4m3_bf16(%v: vector<4xf8E4M3FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E4M3FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e4m3_f32
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E4M3FN> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.fp8 [[BITCAST]][false], %arg1 : vector<2xf32>
func.func @scaled_ext_half_f8e4m3_f32(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e4m3_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E4M3FN> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.fp8 [[BITCAST]][false], %arg1 : vector<2xf16>
func.func @scaled_ext_half_f8e4m3_f16(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e4m3_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E4M3FN> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.fp8 [[BITCAST]][false], %arg1 : vector<2xbf16>
func.func @scaled_ext_half_f8e4m3_bf16(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e4m3_f32
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E4M3FN> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.fp8 [[BITCAST]][false], %arg1 : vector<2xf32>
func.func @scaled_ext_scalar_f8e4m3_f32(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e4m3_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E4M3FN> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.fp8 [[BITCAST]][false], %arg1 : vector<2xf16>
func.func @scaled_ext_scalar_f8e4m3_f16(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e4m3_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E4M3FN> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.fp8 [[BITCAST]][false], %arg1 : vector<2xbf16>
func.func @scaled_ext_scalar_f8e4m3_bf16(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e5m2_f32
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.bf8 [[BITCAST]][false], %arg1 : vector<2xf32>
func.func @scaled_ext_full_f8e5m2_f32(%v: vector<4xf8E5M2>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E5M2> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e5m2_f16
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.bf8 [[BITCAST]][false], %arg1 : vector<2xf16>
func.func @scaled_ext_full_f8e5m2_f16(%v: vector<4xf8E5M2>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E5M2> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e5m2_bf16
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<4xf8E5M2> to vector<4xi8>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.bf8 [[BITCAST]][false], %arg1 : vector<2xbf16>
func.func @scaled_ext_full_f8e5m2_bf16(%v: vector<4xf8E5M2>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E5M2> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e5m2_f32
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E5M2> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.bf8 [[BITCAST]][false], %arg1 : vector<2xf32>
func.func @scaled_ext_half_f8e5m2_f32(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e5m2_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E5M2> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.bf8 [[BITCAST]][false], %arg1 : vector<2xf16>
func.func @scaled_ext_half_f8e5m2_f16(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e5m2_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E5M2> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.bf8 [[BITCAST]][false], %arg1 : vector<2xbf16>
func.func @scaled_ext_half_f8e5m2_bf16(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e5m2_f32
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E5M2> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.bf8 [[BITCAST]][false], %arg1 : vector<2xf32>
func.func @scaled_ext_scalar_f8e5m2_f32(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e5m2_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E5M2> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.bf8 [[BITCAST]][false], %arg1 : vector<2xf16>
func.func @scaled_ext_scalar_f8e5m2_f16(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e5m2_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf8E5M2> to vector<2xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi8>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.bf8 [[BITCAST]][false], %arg1 : vector<2xbf16>
func.func @scaled_ext_scalar_f8e5m2_bf16(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f4e2m1_f32
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.fp4 [[BITCAST]][0], %arg1 : vector<2xf32>
func.func @scaled_ext_full_f4e2m1_f32(%v: vector<8xf4E2M1FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<8xf4E2M1FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_full_f4e2m1_f16
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.fp4 [[BITCAST]][0], %arg1 : vector<2xf16>
func.func @scaled_ext_full_f4e2m1_f16(%v: vector<8xf4E2M1FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<8xf4E2M1FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f4e2m1_bf16
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.fp4 [[BITCAST]][0], %arg1 : vector<2xbf16>
func.func @scaled_ext_full_f4e2m1_bf16(%v: vector<8xf4E2M1FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<8xf4E2M1FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f4e2m1_f32
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<8xf4E2M1FN> to vector<8xi4>
// CHECK-DAG:   [[BITCAST:%.+]] = llvm.bitcast [[CAST]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.fp4 [[BITCAST]][0], %arg1 : vector<2xf32>
func.func @scaled_ext_half_f4e2m1_f32(%v: vector<8xf4E2M1FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<8xf4E2M1FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_half_f4e2m1_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<4xf4E2M1FN> to vector<4xi4>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<8xi4>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<4xi4>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<4xi4>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C2:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:       [[ELEM_2:%.+]] = llvm.extractelement [[V]]{{\[}}[[C2]] : i32] : vector<4xi4>
// CHECK:       [[VEC_2:%.+]] = llvm.insertelement [[ELEM_2]], [[VEC_1]]{{\[}}[[C2]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C3:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:       [[ELEM_3:%.+]] = llvm.extractelement [[V]]{{\[}}[[C3]] : i32] : vector<4xi4>
// CHECK:       [[VEC_3:%.+]] = llvm.insertelement [[ELEM_3]], [[VEC_2]]{{\[}}[[C3]] : i32] : vector<8xi4>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_3]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.fp4 [[BITCAST]][0], %arg1 : vector<2xf16>
func.func @scaled_ext_half_f4e2m1_f16(%v: vector<4xf4E2M1FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf4E2M1FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f4e2m1_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<4xf4E2M1FN> to vector<4xi4>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<8xi4>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<4xi4>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<4xi4>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C2:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:       [[ELEM_2:%.+]] = llvm.extractelement [[V]]{{\[}}[[C2]] : i32] : vector<4xi4>
// CHECK:       [[VEC_2:%.+]] = llvm.insertelement [[ELEM_2]], [[VEC_1]]{{\[}}[[C2]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C3:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:       [[ELEM_3:%.+]] = llvm.extractelement [[V]]{{\[}}[[C3]] : i32] : vector<4xi4>
// CHECK:       [[VEC_3:%.+]] = llvm.insertelement [[ELEM_3]], [[VEC_2]]{{\[}}[[C3]] : i32] : vector<8xi4>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_3]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.fp4 [[BITCAST]][0], %arg1 : vector<2xbf16>
func.func @scaled_ext_half_f4e2m1_bf16(%v: vector<4xf4E2M1FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf4E2M1FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f4e2m1_f32
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf4E2M1FN> to vector<2xi4>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<8xi4>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi4>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi4>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<8xi4>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.fp4 [[BITCAST]][0], %arg1 : vector<2xf32>
func.func @scaled_ext_scalar_f4e2m1_f32(%v: vector<2xf4E2M1FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf4E2M1FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f4e2m1_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf4E2M1FN> to vector<2xi4>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<8xi4>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi4>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi4>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<8xi4>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.fp4 [[BITCAST]][0], %arg1 : vector<2xf16>
func.func @scaled_ext_scalar_f4e2m1_f16(%v: vector<2xf4E2M1FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf4E2M1FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f4e2m1_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<2xf4E2M1FN> to vector<2xi4>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<8xi4>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi4>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<8xi4>
// CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1]] : i32] : vector<2xi4>
// CHECK:       [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1]] : i32] : vector<8xi4>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.fp4 [[BITCAST]][0], %arg1 : vector<2xbf16>
func.func @scaled_ext_scalar_f4e2m1_bf16(%v: vector<2xf4E2M1FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf4E2M1FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_one_f8e4m3_f32
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf8E4M3FN> to vector<1xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.fp8 [[BITCAST]][false], %arg1 : vector<2xf32>
func.func @scaled_ext_one_f8e4m3_f32(%v: vector<1xf8E4M3FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_one_f8e4m3_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf8E4M3FN> to vector<1xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.fp8 [[BITCAST]][false], %arg1 : vector<2xf16>
func.func @scaled_ext_one_f8e4m3_f16(%v: vector<1xf8E4M3FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf8E4M3FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_one_f8e4m3_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf8E4M3FN> to vector<1xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.fp8 [[BITCAST]][false], %arg1 : vector<2xbf16>
func.func @scaled_ext_one_f8e4m3_bf16(%v: vector<1xf8E4M3FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf8E4M3FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_one_f8e5m2_f32
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf8E5M2> to vector<1xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.bf8 [[BITCAST]][false], %arg1 : vector<2xf32>
func.func @scaled_ext_one_f8e5m2_f32(%v: vector<1xf8E5M2>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf8E5M2> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_one_f8e5m2_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf8E5M2> to vector<1xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.bf8 [[BITCAST]][false], %arg1 : vector<2xf16>
func.func @scaled_ext_one_f8e5m2_f16(%v: vector<1xf8E5M2>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf8E5M2> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_one_f8e5m2_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf8E5M2> to vector<1xi8>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<4xi8>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi8>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<4xi8> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.bf8 [[BITCAST]][false], %arg1 : vector<2xbf16>
func.func @scaled_ext_one_f8e5m2_bf16(%v: vector<1xf8E5M2>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf8E5M2> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_one_f4e2m1_f32
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf4E2M1FN> to vector<1xi4>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<8xi4>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi4>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<8xi4>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f32.fp4 [[BITCAST]][0], %arg1 : vector<2xf32>
func.func @scaled_ext_one_f4e2m1_f32(%v: vector<1xf4E2M1FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf4E2M1FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_one_f4e2m1_f16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf4E2M1FN> to vector<1xi4>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<8xi4>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi4>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<8xi4>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.f16.fp4 [[BITCAST]][0], %arg1 : vector<2xf16>
func.func @scaled_ext_one_f4e2m1_f16(%v: vector<1xf4E2M1FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf4E2M1FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_one_f4e2m1_bf16
// CHECK:       [[V:%.+]] = builtin.unrealized_conversion_cast %arg0 : vector<1xf4E2M1FN> to vector<1xi4>
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : vector<8xi4>
// CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<1xi4>
// CHECK:       [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[ZERO]]{{\[}}[[C0]] : i32] : vector<8xi4>
// CHECK:       [[BITCAST:%.+]] = llvm.bitcast [[VEC_0]] : vector<8xi4> to i32
// CHECK:       rocdl.cvt.scalef32.pk.bf16.fp4 [[BITCAST]][0], %arg1 : vector<2xbf16>
func.func @scaled_ext_one_f4e2m1_bf16(%v: vector<1xf4E2M1FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<1xf4E2M1FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}
