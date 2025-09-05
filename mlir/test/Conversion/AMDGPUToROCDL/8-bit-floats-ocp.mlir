// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1200 | FileCheck %s

// CHECK-LABEL: func @ext_scalar
// CHECK: [[V:%.+]] = builtin.unrealized_conversion_cast %{{.+}} : f8E5M2 to i8
// CHECK-DAG: [[UNDEF:%.+]] = llvm.mlir.undef : vector<4xi8>
// CHECK-DAG: [[C0_1:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[VEC:%.+]] = llvm.insertelement [[V]], [[UNDEF]]{{\[}}[[C0_1]] : i32] : vector<4xi8>
// CHECK: [[CAST:%.+]] = llvm.bitcast [[VEC]] : vector<4xi8> to i32
// CHECK: [[EXT:%.+]] = rocdl.cvt.f32.bf8 [[CAST]][0] : f32
// CHECK: return [[EXT]] : f32
func.func @ext_scalar(%v: f8E5M2) -> f32 {
  %ret = amdgpu.ext_packed_fp8 %v[0] : f8E5M2 to f32
  func.return %ret : f32
}

// CHECK-LABEL: func @ext_short_vec
// CHECK: [[V:%.+]] = builtin.unrealized_conversion_cast %{{.+}} : vector<2xf8E4M3FN> to vector<2xi8>
// CHECK-DAG: [[UNDEF:%.+]] = llvm.mlir.undef : vector<4xi8>
// CHECK-DAG: [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK: [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[UNDEF]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK: [[C1_1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1_1]] : i32] : vector<2xi8>
// CHECK: [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1_1]] : i32] : vector<4xi8>
// CHECK: [[CAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK: [[EXT:%.+]] = rocdl.cvt.f32.fp8 [[CAST]][1] : f32
// CHECK: return [[EXT]] : f32
func.func @ext_short_vec(%v: vector<2xf8E4M3FN>) -> f32 {
  %ret = amdgpu.ext_packed_fp8 %v[1] : vector<2xf8E4M3FN> to f32
  func.return %ret : f32
}

// CHECK-LABEL: func @ext_full_vec(
// CHECK: [[V:%.+]] = builtin.unrealized_conversion_cast %{{.+}} : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK: [[CAST:%.+]] = llvm.bitcast [[V]] : vector<4xi8> to i32
// CHECK: [[EXT:%.+]] = rocdl.cvt.f32.fp8 [[CAST]][3] : f32
// CHECK: return [[EXT]] : f32
func.func @ext_full_vec(%v: vector<4xf8E4M3FN>) -> f32 {
  %ret = amdgpu.ext_packed_fp8 %v[3] : vector<4xf8E4M3FN> to f32
  func.return %ret : f32
}

// CHECK-LABEL: func @ext_packed_2xfp8
// CHECK: [[V:%.+]] = builtin.unrealized_conversion_cast %{{.+}} : vector<2xf8E4M3FN> to vector<2xi8>
// CHECK-DAG: [[UNDEF:%.+]] = llvm.mlir.undef : vector<4xi8>
// CHECK-DAG: [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ELEM_0:%.+]] = llvm.extractelement [[V]]{{\[}}[[C0]] : i32] : vector<2xi8>
// CHECK: [[VEC_0:%.+]] = llvm.insertelement [[ELEM_0]], [[UNDEF]]{{\[}}[[C0]] : i32] : vector<4xi8>
// CHECK: [[C1_1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: [[ELEM_1:%.+]] = llvm.extractelement [[V]]{{\[}}[[C1_1]] : i32] : vector<2xi8>
// CHECK: [[VEC_1:%.+]] = llvm.insertelement [[ELEM_1]], [[VEC_0]]{{\[}}[[C1_1]] : i32] : vector<4xi8>
// CHECK: [[CAST:%.+]] = llvm.bitcast [[VEC_1]] : vector<4xi8> to i32
// CHECK: [[EXT:%.+]] = rocdl.cvt.pk.f32.fp8 [[CAST]][false] : vector<2xf32>
// CHECK: return [[EXT]]
func.func @ext_packed_2xfp8(%v: vector<2xf8E4M3FN>) -> vector<2xf32> {
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<2xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func @ext_packed_4xfp8
// CHECK: [[V:%.+]] = builtin.unrealized_conversion_cast %{{.+}} : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK: [[CAST:%.+]] = llvm.bitcast [[V]] : vector<4xi8> to i32
// CHECK: [[EXT:%.+]] = rocdl.cvt.pk.f32.fp8 [[CAST]][true] : vector<2xf32>
// CHECK: return [[EXT]] : vector<2xf32>
func.func @ext_packed_4xfp8(%v: vector<4xf8E4M3FN>) -> vector<2xf32> {
  %ret = amdgpu.ext_packed_fp8 %v[1] : vector<4xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func @packed_trunc
// CHECK-SAME: ([[V:%.+]]: f32)
// CHECK: [[V2:%.+]] = llvm.mlir.undef : f32
// CHECK: [[EXISTING:%.+]] = llvm.mlir.undef : i32
// CHECK: [[PACKED:%.+]] = rocdl.cvt.pk.fp8.f32 [[V]], [[V2]] -> [[EXISTING]][false] : i32
// CHECK: [[CAST:%.+]] = llvm.bitcast [[PACKED]] : i32 to vector<4xi8>
// CHECK: builtin.unrealized_conversion_cast [[CAST]] : vector<4xi8> to vector<4xf8E4M3FN>
func.func @packed_trunc(%v: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_trunc_2xfp8 %v, undef into undef[word 0] : f32 to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func @packed_truncx2
// CHECK-SAME: ([[V:%.+]]: f32, [[W:%.+]]: f32)
// CHECK: [[EXISTING:%.+]] = llvm.mlir.undef : i32
// CHECK: [[PACKED:%.+]] = rocdl.cvt.pk.fp8.f32 [[V]], [[W]] -> [[EXISTING]][false] : i32
// CHECK: [[CAST:%.+]] = llvm.bitcast [[PACKED]] : i32 to vector<4xi8>
// CHECK: builtin.unrealized_conversion_cast [[CAST]] : vector<4xi8> to vector<4xf8E4M3FN>
func.func @packed_truncx2(%v: f32, %w: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_trunc_2xfp8 %v, %w into undef[word 0] : f32 to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func @packed_truncx2_into
// CHECK-SAME: ([[V:%.+]]: f32, [[W:%.+]]: f32, [[EXISTING:%.+]]:  vector<4xf8E5M2>)
// CHECK: [[EXISTING_BYTES:%.+]] = builtin.unrealized_conversion_cast [[EXISTING]] : vector<4xf8E5M2> to vector<4xi8>
// CHECK: [[EXISTING_INT:%.+]] = llvm.bitcast [[EXISTING_BYTES]] : vector<4xi8> to i32
// CHECK: [[PACKED:%.+]] = rocdl.cvt.pk.bf8.f32 [[V]], [[W]] -> [[EXISTING_INT]][true] : i32
// CHECK: [[CAST:%.+]] = llvm.bitcast [[PACKED]] : i32 to vector<4xi8>
// CHECK: builtin.unrealized_conversion_cast [[CAST]] : vector<4xi8> to vector<4xf8E5M2>
func.func @packed_truncx2_into(%v: f32, %w: f32, %existing: vector<4xf8E5M2>) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_trunc_2xfp8 %v, %w into %existing[word 1] : f32 to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func @packed_stoch_round
// CHECK-SAME: ([[V:%.+]]: f32, [[S:%.+]]: i32)
// CHECK: [[EXISTING:%.+]] = llvm.mlir.undef : i32
// CHECK: [[PACKED:%.+]] = rocdl.cvt.sr.fp8.f32 [[V]], [[S]] -> [[EXISTING]][0] : i32
// CHECK: [[CAST:%.+]] = llvm.bitcast [[PACKED]] : i32 to vector<4xi8>
// CHECK:  builtin.unrealized_conversion_cast [[CAST]] : vector<4xi8> to vector<4xf8E4M3FN>
func.func @packed_stoch_round(%v: f32, %s: i32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_stoch_round_fp8 %v + %s into undef[0] : f32 to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func @packed_stoch_round_into
// CHECK-SAME: ([[V:%.+]]: f32, [[S:%.+]]: i32, [[EXISTING:%.+]]:  vector<4xf8E5M2>)
// CHECK: [[EXISTING_BYTES:%.+]] = builtin.unrealized_conversion_cast [[EXISTING]] : vector<4xf8E5M2> to vector<4xi8>
// CHECK: [[EXISTING_INT:%.+]] = llvm.bitcast [[EXISTING_BYTES]] : vector<4xi8> to i32
// CHECK: [[PACKED:%.+]] = rocdl.cvt.sr.bf8.f32 [[V]], [[S]] -> [[EXISTING_INT]][1] : i32
// CHECK: [[CAST:%.+]] = llvm.bitcast [[PACKED]] : i32 to vector<4xi8>
// CHECK: builtin.unrealized_conversion_cast [[CAST]] : vector<4xi8> to vector<4xf8E5M2>
func.func @packed_stoch_round_into(%v: f32, %s: i32, %existing: vector<4xf8E5M2>) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_stoch_round_fp8 %v + %s into %existing[1] : f32 to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}
