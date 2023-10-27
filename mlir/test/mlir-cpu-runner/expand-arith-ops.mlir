// RUN:   mlir-opt %s -pass-pipeline="builtin.module(func.func(arith-expand{include-bf16=true},convert-arith-to-llvm),convert-vector-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner                                                      \
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
  // CHECK: 1.00781
  %roundOneI = arith.constant 0x3f808000 : i32
  %roundOneF = arith.bitcast %roundOneI : i32 to f32
  call @trunc_bf16(%roundOneF): (f32) -> ()

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

  // CHECK-NEXT: 3.38953e+38
  %bigi = arith.constant 0x7f7fffff : i32
  %bigf = arith.bitcast %bigi : i32 to f32
  call @trunc_bf16(%bigf): (f32) -> ()

  // CHECK-NEXT: -3.38953e+38
  %negbigi = arith.constant 0xff7fffff : i32
  %negbigf = arith.bitcast %negbigi : i32 to f32
  call @trunc_bf16(%negbigf): (f32) -> ()

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
