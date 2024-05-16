// RUN: mlir-opt %s --convert-libc-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm | mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

module {
  func.func @doprint(%t: f32, %t2: i32, %t3: i64) {
    libc.printf "Hello world %f %d %lld\n" %t, %t2, %t3 : f32, i32, i64
    return
  }

  func.func @main() {
    %c2 = arith.constant 2.0 : f32
    %c32i = arith.constant 2000000 : i32
    %c64i = arith.constant 2000000 : i64
    call @doprint(%c2, %c32i, %c64i) : (f32, i32, i64) -> ()
    return
  }
  // CHECK: Hello world 2.000000 2000000 2000000
}