// RUN: mlir-opt %s --arith-expand="include-f8e8m0=true" --canonicalize | FileCheck %s

// The f16 -> f8E8M0FNU expansion widens %h to f32 and reads the exponent from
// its bits. If the widening kept 'contract', ExtFOp::fold would cancel it
// against the contract truncf that produced %h, and the exponent would come
// from %x before its rounding to f16: f32 2 - 2^-11 rounds to 2.0 in f16, so it
// must give 2^1, not 2^0.

// CHECK-LABEL: @truncf_f8E8M0FNU_after_contract_truncf
// CHECK: %[[H:.+]] = arith.truncf %{{.+}} fastmath<contract> : f32 to f16
// CHECK: arith.extf %[[H]] : f16 to f32
func.func @truncf_f8E8M0FNU_after_contract_truncf(%x: f32) -> f8E8M0FNU {
  %h = arith.truncf %x fastmath<contract> : f32 to f16
  %e = arith.truncf %h fastmath<contract> : f16 to f8E8M0FNU
  return %e : f8E8M0FNU
}
