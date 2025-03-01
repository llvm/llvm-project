// RUN:   mlir-opt %s -pass-pipeline="builtin.module(func.func(arith-expand{include-bf16=true},convert-arith-to-llvm),convert-vector-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_c_runner_utils  \
// RUN:     -shared-libs=%mlir_runner_utils    \
// RUN: | FileCheck %s

func.func @trunc_bf16(%a : f32) {
  %b = arith.truncf %a : f32 to bf16
  %c = arith.extf %b : bf16 to f32
  vector.print %c : f32
  return
}

func.func @main() {
  // Note: this is a tie (low 16 bits are 0x8000). We expect the rounding behavior
  // to break ties "to nearest-even", which in this case means downwards,
  // since bit 16 is not set.
  // CHECK: 1
  %value_1_00391_I = arith.constant 0x3f808000 : i32
  %value_1_00391_F = arith.bitcast %value_1_00391_I : i32 to f32
  call @trunc_bf16(%value_1_00391_F): (f32) -> ()

  // Note: this is a tie (low 16 bits are 0x8000). We expect the rounding behavior
  // to break ties "to nearest-even", which in this case means upwards,
  // since bit 16 is set.
  // CHECK-NEXT: 1.0156
  %value_1_01172_I = arith.constant 0x3f818000 : i32
  %value_1_01172_F = arith.bitcast %value_1_01172_I : i32 to f32
  call @trunc_bf16(%value_1_01172_F): (f32) -> ()

  // CHECK-NEXT: -1
  %noRoundNegOneI = arith.constant 0xbf808000 : i32
  %noRoundNegOneF = arith.bitcast %noRoundNegOneI : i32 to f32
  call @trunc_bf16(%noRoundNegOneF): (f32) -> ()

  // CHECK-NEXT: -1.00781
  %roundNegOneI = arith.constant 0xbf808001 : i32
  %roundNegOneF = arith.bitcast %roundNegOneI : i32 to f32
  call @trunc_bf16(%roundNegOneF): (f32) -> ()

  // CHECK-NEXT: inf
  %infi = arith.constant 0x7f800000 : i32
  %inff = arith.bitcast %infi : i32 to f32
  call @trunc_bf16(%inff): (f32) -> ()

  // CHECK-NEXT: -inf
  %neginfi = arith.constant 0xff800000 : i32
  %neginff = arith.bitcast %neginfi : i32 to f32
  call @trunc_bf16(%neginff): (f32) -> ()

  // Note: this rounds upwards. As the mantissa was already saturated, this rounding
  // causes the exponent to be incremented. As the exponent was already the
  // maximum exponent value for finite values, this increment of the exponent
  // causes this to overflow to +inf.
  // CHECK-NEXT: inf
  %big_overflowing_i = arith.constant 0x7f7fffff : i32
  %big_overflowing_f = arith.bitcast %big_overflowing_i : i32 to f32
  call @trunc_bf16(%big_overflowing_f): (f32) -> ()

  // Same as the previous testcase but negative.
  // CHECK-NEXT: -inf
  %negbig_overflowing_i = arith.constant 0xff7fffff : i32
  %negbig_overflowing_f = arith.bitcast %negbig_overflowing_i : i32 to f32
  call @trunc_bf16(%negbig_overflowing_f): (f32) -> ()

  // In contrast to the previous two testcases, the upwards-rounding here
  // does not cause overflow.
  // CHECK-NEXT: 3.38953e+38
  %big_nonoverflowing_i = arith.constant 0x7f7effff : i32
  %big_nonoverflowing_f = arith.bitcast %big_nonoverflowing_i : i32 to f32
  call @trunc_bf16(%big_nonoverflowing_f): (f32) -> ()

  // CHECK-NEXT: 1.625
  %exprolli = arith.constant 0x3fcfffff : i32
  %exprollf = arith.bitcast %exprolli : i32 to f32
  call @trunc_bf16(%exprollf): (f32) -> ()

  // CHECK-NEXT: -1.625
  %exprollnegi = arith.constant 0xbfcfffff : i32
  %exprollnegf = arith.bitcast %exprollnegi : i32 to f32
  call @trunc_bf16(%exprollnegf): (f32) -> ()

  return
}
