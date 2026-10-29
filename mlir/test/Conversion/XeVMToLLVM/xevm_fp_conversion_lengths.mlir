// RUN: mlir-opt --convert-xevm-to-llvm --split-input-file %s | FileCheck %s

// The fp8 and fp4 conversions are provided as one builtin per vector length.
// This file covers the SPIR-V vector lengths other than 16, which
// xevm_mx-to-llvm.mlir covers.

// CHECK: llvm.func spir_funccc @__builtin_IB_hftobf8_2(vector<2xf16>) -> vector<2xi8>
// CHECK-LABEL: llvm.func @truncf_f16_to_bf8_v2
// CHECK-SAME: %[[ARG0:.*]]: vector<2xf16>
llvm.func @truncf_f16_to_bf8_v2(%src: vector<2xf16>) -> vector<2xi8> {
  // CHECK: %[[RES:.*]] = llvm.call spir_funccc @__builtin_IB_hftobf8_2(%[[ARG0]])
  // CHECK-SAME: : (vector<2xf16>) -> vector<2xi8>
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = bf8 } : (vector<2xf16>) -> vector<2xi8>
  llvm.return %dst : vector<2xi8>
}

// -----

// A 3 element vector is a SPIR-V vector length, and the builtins provide it.
// CHECK: llvm.func spir_funccc @__builtin_IB_hftohf8_3(vector<3xf16>) -> vector<3xi8>
// CHECK: llvm.func spir_funccc @_Z13convert_half3Dv3_f(vector<3xf32>) -> vector<3xf16>
// CHECK: llvm.func spir_funccc @__builtin_IB_bftof_3(vector<3xi16>) -> vector<3xf32>
// CHECK-LABEL: llvm.func @truncf_bf16_to_hf8_v3
// CHECK-SAME: %[[ARG0:.*]]: vector<3xbf16>
llvm.func @truncf_bf16_to_hf8_v3(%src: vector<3xbf16>) -> vector<3xi8> {
  // CHECK: %[[BC:.*]] = llvm.bitcast %[[ARG0]] : vector<3xbf16> to vector<3xi16>
  // CHECK: %[[F32:.*]] = llvm.call spir_funccc @__builtin_IB_bftof_3(%[[BC]])
  // CHECK-SAME: : (vector<3xi16>) -> vector<3xf32>
  // CHECK: %[[F16:.*]] = llvm.call spir_funccc @_Z13convert_half3Dv3_f(%[[F32]])
  // CHECK-SAME: : (vector<3xf32>) -> vector<3xf16>
  // CHECK: %[[RES:.*]] = llvm.call spir_funccc @__builtin_IB_hftohf8_3(%[[F16]])
  // CHECK-SAME: : (vector<3xf16>) -> vector<3xi8>
  %dst = xevm.truncf %src { src_etype = bf16, dst_etype = f8 } : (vector<3xbf16>) -> vector<3xi8>
  llvm.return %dst : vector<3xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_hftobf8_8(vector<8xf16>) -> vector<8xi8>
// CHECK-LABEL: llvm.func @truncf_f16_to_bf8_v8
llvm.func @truncf_f16_to_bf8_v8(%src: vector<8xf16>) -> vector<8xi8> {
  // CHECK: llvm.call spir_funccc @__builtin_IB_hftobf8_8({{.*}}) {{.*}} : (vector<8xf16>) -> vector<8xi8>
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = bf8 } : (vector<8xf16>) -> vector<8xi8>
  llvm.return %dst : vector<8xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_bf8tohf_2(vector<2xi8>) -> vector<2xf16>
// CHECK-LABEL: llvm.func @extf_bf8_to_f16_v2
llvm.func @extf_bf8_to_f16_v2(%src: vector<2xi8>) -> vector<2xf16> {
  // CHECK: llvm.call spir_funccc @__builtin_IB_bf8tohf_2({{.*}}) {{.*}} : (vector<2xi8>) -> vector<2xf16>
  %dst = xevm.extf %src { src_etype = bf8, dst_etype = f16 } : (vector<2xi8>) -> vector<2xf16>
  llvm.return %dst : vector<2xf16>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_ftobf_4(vector<4xf32>) -> vector<4xi16>
// CHECK: llvm.func spir_funccc @_Z14convert_float4Dv4_Dh(vector<4xf16>) -> vector<4xf32>
// CHECK: llvm.func spir_funccc @__builtin_IB_bf8tohf_4(vector<4xi8>) -> vector<4xf16>
// CHECK-LABEL: llvm.func @extf_bf8_to_bf16_v4
// CHECK-SAME: %[[ARG0:.*]]: vector<4xi8>
llvm.func @extf_bf8_to_bf16_v4(%src: vector<4xi8>) -> vector<4xbf16> {
  // CHECK: %[[F16:.*]] = llvm.call spir_funccc @__builtin_IB_bf8tohf_4(%[[ARG0]])
  // CHECK-SAME: : (vector<4xi8>) -> vector<4xf16>
  // CHECK: %[[F32:.*]] = llvm.call spir_funccc @_Z14convert_float4Dv4_Dh(%[[F16]])
  // CHECK-SAME: : (vector<4xf16>) -> vector<4xf32>
  // CHECK: %[[BF:.*]] = llvm.call spir_funccc @__builtin_IB_ftobf_4(%[[F32]])
  // CHECK-SAME: : (vector<4xf32>) -> vector<4xi16>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[BF]] : vector<4xi16> to vector<4xbf16>
  %dst = xevm.extf %src { src_etype = bf8, dst_etype = bf16 } : (vector<4xi8>) -> vector<4xbf16>
  llvm.return %dst : vector<4xbf16>
}

// -----

// Two fp4 values pack into a single byte. SPIR-V has no vector of length one,
// so the packed side is a scalar, and only the low byte of the dnscl result is
// kept.
// CHECK: llvm.func spir_funccc @__builtin_IB_dnscl_hf16(i32, i32, i32, i32) -> i32
// CHECK-LABEL: llvm.func @truncf_f16_to_e2m1_v2
// CHECK-SAME: %[[ARG0:.*]]: vector<2xf16>
llvm.func @truncf_f16_to_e2m1_v2(%src: vector<2xf16>) -> i8 {
  // CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : i32
  // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[BC:.*]] = llvm.bitcast %[[ARG0]] : vector<2xf16> to i32
  // CHECK: %[[CALL:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_hf16(%[[BC]], %[[UNDEF]], %[[C1]], %[[C0]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[RES:.*]] = llvm.trunc %[[CALL]] : i32 to i8
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = e2m1 } : (vector<2xf16>) -> i8
  llvm.return %dst : i8
}

// -----

// An element pair is the conversion granularity, so 3 elements are padded up to
// 4 and the spare nibble is left undefined. The two bytes the call writes, 0
// and 2, are compacted into the result.
// CHECK: llvm.func spir_funccc @__builtin_IB_dnscl_hf16(i32, i32, i32, i32) -> i32
// CHECK-LABEL: llvm.func @truncf_f16_to_e2m1_v3
// CHECK-SAME: %[[ARG0:.*]]: vector<3xf16>
llvm.func @truncf_f16_to_e2m1_v3(%src: vector<3xf16>) -> vector<2xi8> {
  // CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[PAD:.*]] = llvm.shufflevector %[[ARG0]], %[[ARG0]] [0, 1, 2, 0] : vector<3xf16>
  // CHECK: %[[LANES:.*]] = llvm.bitcast %[[PAD]] : vector<4xf16> to vector<2xi32>
  // CHECK-DAG: %[[L1:.*]] = llvm.extractelement %[[LANES]][%[[C1]] : i32] : vector<2xi32>
  // CHECK-DAG: %[[L0:.*]] = llvm.extractelement %[[LANES]][%[[C0]] : i32] : vector<2xi32>
  // CHECK: %[[CALL:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_hf16(%[[L0]], %[[L1]], %[[C1]], %[[C0]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[BYTES:.*]] = llvm.bitcast %[[CALL]] : i32 to vector<4xi8>
  // CHECK: %[[RES:.*]] = llvm.shufflevector %[[BYTES]], %[[BYTES]] [0, 2] : vector<4xi8>
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = e2m1 } : (vector<3xf16>) -> vector<2xi8>
  llvm.return %dst : vector<2xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_dnscl_hf16(i32, i32, i32, i32) -> i32
// CHECK-LABEL: llvm.func @truncf_f16_to_e2m1_v4
// CHECK-SAME: %[[ARG0:.*]]: vector<4xf16>
llvm.func @truncf_f16_to_e2m1_v4(%src: vector<4xf16>) -> vector<2xi8> {
  // CHECK: %[[LANES:.*]] = llvm.bitcast %[[ARG0]] : vector<4xf16> to vector<2xi32>
  // CHECK: llvm.call spir_funccc @__builtin_IB_dnscl_hf16
  // CHECK: %[[BYTES:.*]] = llvm.bitcast {{.*}} : i32 to vector<4xi8>
  // CHECK: %[[RES:.*]] = llvm.shufflevector %[[BYTES]], %[[BYTES]] [0, 2] : vector<4xi8>
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = e2m1 } : (vector<4xf16>) -> vector<2xi8>
  llvm.return %dst : vector<2xi8>
}

// -----

// Eight elements fill a dword: two calls with complementary nibble modes, 0 for
// bytes 0 and 2 and 2 for bytes 1 and 3, are OR-ed together.
// CHECK: llvm.func spir_funccc @__builtin_IB_dnscl_bf16(i32, i32, i32, i32) -> i32
// CHECK-LABEL: llvm.func @truncf_bf16_to_e2m1_v8
// CHECK-SAME: %[[ARG0:.*]]: vector<8xbf16>
llvm.func @truncf_bf16_to_e2m1_v8(%src: vector<8xbf16>) -> vector<4xi8> {
  // CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[C3:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[LANES:.*]] = llvm.bitcast %[[ARG0]] : vector<8xbf16> to vector<4xi32>
  // CHECK: %[[L0:.*]] = llvm.extractelement %[[LANES]][%[[C0]] : i32] : vector<4xi32>
  // CHECK: %[[L2:.*]] = llvm.extractelement %[[LANES]][%[[C2]] : i32] : vector<4xi32>
  // CHECK: %[[EVEN:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_bf16(%[[L0]], %[[L2]], %[[C1]], %[[C0]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[L1:.*]] = llvm.extractelement %[[LANES]][%[[C1]] : i32] : vector<4xi32>
  // CHECK: %[[L3:.*]] = llvm.extractelement %[[LANES]][%[[C3]] : i32] : vector<4xi32>
  // CHECK: %[[ODD:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_bf16(%[[L1]], %[[L3]], %[[C1]], %[[C2]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[OR:.*]] = llvm.or %[[EVEN]], %[[ODD]] : i32
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[OR]] : i32 to vector<4xi8>
  %dst = xevm.truncf %src { src_etype = bf16, dst_etype = e2m1 } : (vector<8xbf16>) -> vector<4xi8>
  llvm.return %dst : vector<4xi8>
}

// -----

// The fp4 up-conversion takes a scalar source byte for the shortest length.
// CHECK: llvm.func spir_funccc @__builtin_IB_shfl_idx4_to_fp16_packed(vector<16xi32>, i8) -> i32
// CHECK: llvm.func spir_funccc @__builtin_IB_shfl_idx4_lut(i32) -> vector<16xi32>
// CHECK-LABEL: llvm.func @extf_e2m1_to_f16_v2
// CHECK-SAME: %[[ARG0:.*]]: i8
llvm.func @extf_e2m1_to_f16_v2(%src: i8) -> vector<2xf16> {
  // CHECK: %[[LUTIDX:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK: %[[LUT:.*]] = llvm.call spir_funccc @__builtin_IB_shfl_idx4_lut(%[[LUTIDX]])
  // CHECK-SAME: : (i32) -> vector<16xi32>
  // CHECK: %[[CONV:.*]] = llvm.call spir_funccc @__builtin_IB_shfl_idx4_to_fp16_packed(%[[LUT]], %[[ARG0]])
  // CHECK-SAME: : (vector<16xi32>, i8) -> i32
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[CONV]] : i32 to vector<2xf16>
  %dst = xevm.extf %src { src_etype = e2m1, dst_etype = f16 } : (i8) -> vector<2xf16>
  llvm.return %dst : vector<2xf16>
}

// -----

// Three values are read as two whole bytes, and the spare value dropped.
// CHECK: llvm.func spir_funccc @__builtin_IB_shfl_idx4_to_fp16_2_packed(vector<16xi32>, vector<2xi8>) -> vector<2xi32>
// CHECK: llvm.func spir_funccc @__builtin_IB_shfl_idx4_lut(i32) -> vector<16xi32>
// CHECK-LABEL: llvm.func @extf_e2m1_to_f16_v3
// CHECK-SAME: %[[ARG0:.*]]: vector<2xi8>
llvm.func @extf_e2m1_to_f16_v3(%src: vector<2xi8>) -> vector<3xf16> {
  // CHECK: %[[CONV:.*]] = llvm.call spir_funccc @__builtin_IB_shfl_idx4_to_fp16_2_packed({{.*}}, %[[ARG0]])
  // CHECK-SAME: : (vector<16xi32>, vector<2xi8>) -> vector<2xi32>
  // CHECK: %[[WIDE:.*]] = llvm.bitcast %[[CONV]] : vector<2xi32> to vector<4xf16>
  // CHECK: %[[RES:.*]] = llvm.shufflevector %[[WIDE]], %[[WIDE]] [0, 1, 2] : vector<4xf16>
  %dst = xevm.extf %src { src_etype = e2m1, dst_etype = f16 } : (vector<2xi8>) -> vector<3xf16>
  llvm.return %dst : vector<3xf16>
}

// -----

// The lookup table index selects the destination format: 5 for bf16, 7 for f16.
// CHECK: llvm.func spir_funccc @__builtin_IB_shfl_idx4_to_fp16_2_packed(vector<16xi32>, vector<2xi8>) -> vector<2xi32>
// CHECK-LABEL: llvm.func @extf_e2m1_to_bf16_v4
llvm.func @extf_e2m1_to_bf16_v4(%src: vector<2xi8>) -> vector<4xbf16> {
  // CHECK: %[[LUTIDX:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK: llvm.call spir_funccc @__builtin_IB_shfl_idx4_lut(%[[LUTIDX]])
  // CHECK: %[[CONV:.*]] = llvm.call spir_funccc @__builtin_IB_shfl_idx4_to_fp16_2_packed
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[CONV]] : vector<2xi32> to vector<4xbf16>
  %dst = xevm.extf %src { src_etype = e2m1, dst_etype = bf16 } : (vector<2xi8>) -> vector<4xbf16>
  llvm.return %dst : vector<4xbf16>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_shfl_idx4_to_fp16_4_packed(vector<16xi32>, vector<4xi8>) -> vector<4xi32>
// CHECK-LABEL: llvm.func @extf_e2m1_to_f16_v8
// CHECK-SAME: %[[ARG0:.*]]: vector<4xi8>
llvm.func @extf_e2m1_to_f16_v8(%src: vector<4xi8>) -> vector<8xf16> {
  // CHECK: %[[CONV:.*]] = llvm.call spir_funccc @__builtin_IB_shfl_idx4_to_fp16_4_packed({{.*}}, %[[ARG0]])
  // CHECK-SAME: : (vector<16xi32>, vector<4xi8>) -> vector<4xi32>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[CONV]] : vector<4xi32> to vector<8xf16>
  %dst = xevm.extf %src { src_etype = e2m1, dst_etype = f16 } : (vector<4xi8>) -> vector<8xf16>
  llvm.return %dst : vector<8xf16>
}
