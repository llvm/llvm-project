// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Test invalid target architecture (sm_100 instead of sm_100a)
gpu.module @invalid_arch_sm_100 [#nvvm.target<chip = "sm_100">] {
  func.func @convert_rs() {
    %f1 = llvm.mlir.constant(1.0 : f32) : f32
    %f2 = llvm.mlir.constant(2.0 : f32) : f32
    %rbits = llvm.mlir.constant(0x12345678 : i32) : i32
    // expected-error@+1 {{'nvvm.convert.f32x2.to.f16x2' op is not supported on sm_100}}
    %res = nvvm.convert.f32x2.to.f16x2 %f1, %f2, %rbits : vector<2xf16>
    return
  }
}

// -----

// Test that operations require stochastic rounding mode
llvm.func @invalid_rnd_mode_f16x2(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xf16> {
  // expected-error@+1 {{Only RS rounding mode is supported for conversions from f32x2 to f16x2.}}
  %res = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB, %rbits {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf16>
  llvm.return %res : vector<2xf16>
}

// -----

llvm.func @invalid_rnd_mode_bf16x2(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xbf16> {
  // expected-error@+1 {{Only RS rounding mode is supported for conversions from f32x2 to bf16x2.}}
  %res = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB, %rbits {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// -----

// Test invalid destination types for f8x4 (should only accept f8E4M3FN, f8E5M2)
llvm.func @invalid_dst_type_f8x4_e3m4(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // expected-error@+1 {{Only 'f8E4M3FN' and 'f8E5M2' types are supported for conversions from f32x4 to f8x4.}}
  %res = nvvm.convert.f32x4.to.f8x4 %src, %rbits : vector<4xf32> -> vector<4xi8> (f8E3M4)
  llvm.return %res : vector<4xi8>
}

// -----

llvm.func @invalid_dst_type_f8x4_e8m0(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // expected-error@+1 {{Only 'f8E4M3FN' and 'f8E5M2' types are supported for conversions from f32x4 to f8x4.}}
  %res = nvvm.convert.f32x4.to.f8x4 %src, %rbits : vector<4xf32> -> vector<4xi8> (f8E8M0FNU)
  llvm.return %res : vector<4xi8>
}

// -----

// Test invalid destination types for f6x4 (should only accept f6E2M3FN, f6E3M2FN)
llvm.func @invalid_dst_type_f6x4_f8(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // expected-error@+1 {{Only 'f6E2M3FN' and 'f6E3M2FN' types are supported for conversions from f32x4 to f6x4.}}
  %res = nvvm.convert.f32x4.to.f6x4 %src, %rbits : vector<4xf32> -> vector<4xi8> (f8E4M3FN)
  llvm.return %res : vector<4xi8>
}

// -----

// Test invalid destination type for f4x4 (should only accept f4E2M1FN)
llvm.func @invalid_dst_type_f4x4_f6(%src : vector<4xf32>, %rbits : i32) -> i16 {
  // expected-error@+1 {{Only 'f4E2M1FN' type is supported for conversions from f32x4 to f4x4.}}
  %res = nvvm.convert.f32x4.to.f4x4 %src, %rbits : vector<4xf32> -> i16 (f6E2M3FN)
  llvm.return %res : i16
}

// -----

// Test invalid rounding modes for non-stochastic ops
llvm.func @convert_float_to_tf32_rs_not_supported(%src : f32) -> i32 {
  // expected-error @below {{Only {rn,rz,rna} rounding modes supported for ConvertFloatToTF32Op.}}
  %res = nvvm.convert.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rs>}
  llvm.return %res : i32
}

// -----

llvm.func @convert_f32x2_to_f8x2_rs_not_supported(%a : f32, %b : f32) {
  // expected-error @below {{Only RN rounding mode is supported for conversions from f32x2 to 'f8E4M3FN' and 'f8E5M2' types}}
  %res = nvvm.convert.f32x2.to.f8x2 %a, %b {rnd = #nvvm.fp_rnd_mode<rs>, sat = #nvvm.sat_mode<satfinite>} : i16 (f8E4M3FN)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_rs_not_supported(%src : vector<2xbf16>) {
  // expected-error @below {{Only RZ and RP rounding modes are supported for conversions from bf16x2 to f8x2.}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src {rnd = #nvvm.fp_rnd_mode<rs>} : vector<2xbf16> -> i16 (f8E8M0FNU)
  llvm.return
}
