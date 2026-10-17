// RUN: mlir-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(affine-loop-fusion{mode=producer maximal}),func.func(lower-affine),generate-runtime-verification,convert-scf-to-cf,convert-to-llvm)' | \
// RUN: mlir-runner -e main -entry-point-result=i64 \
// RUN:   %if target={{s390x-.*}} %{ -argext-abi-check=false %} | FileCheck %s
// XFAIL: system-aix

// CHECK: {{^0$}}

// The producer overwrites the even elements; the odd elements keep their
// initial values. Runtime verification makes an undersized private buffer fail
// deterministically instead of relying on undefined out-of-bounds behavior.
memref.global "private" constant @expected : memref<16xf64> =
    dense<[1000.0, 101.0, 1001.0, 103.0, 1002.0, 105.0, 1003.0, 107.0,
           1004.0, 109.0, 1005.0, 111.0, 1006.0, 113.0, 1007.0, 115.0]>

func.func @kernel(%in: memref<32xf64>, %comm: memref<32xf64>,
                  %out: memref<32xf64>) {
  affine.for %i = 0 to 8 {
    %a = affine.load %in[%i] : memref<32xf64>
    affine.store %a, %comm[2 * %i] : memref<32xf64>
  }
  affine.for %j = 0 to 16 {
    %b = affine.load %comm[%j] : memref<32xf64>
    affine.store %b, %out[%j] : memref<32xf64>
  }
  return
}

func.func @main() -> i64 {
  %in = memref.alloc() : memref<32xf64>
  %comm = memref.alloc() : memref<32xf64>
  %out = memref.alloc() : memref<32xf64>
  %expected = memref.get_global @expected : memref<16xf64>
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c100 = arith.constant 100.0 : f64
  %c1000 = arith.constant 1000.0 : f64
  %c0Index = arith.constant 0 : index
  %c1Index = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  scf.for %i = %c0Index to %c32 step %c1Index {
    %i64 = arith.index_cast %i : index to i64
    %f64 = arith.sitofp %i64 : i64 to f64
    %inValue = arith.addf %c1000, %f64 : f64
    %commValue = arith.addf %c100, %f64 : f64
    memref.store %inValue, %in[%i] : memref<32xf64>
    memref.store %commValue, %comm[%i] : memref<32xf64>
  }

  call @kernel(%in, %comm, %out)
      : (memref<32xf64>, memref<32xf64>, memref<32xf64>) -> ()

  %errors = scf.for %j = %c0Index to %c16 step %c1Index
      iter_args(%errorsIn = %c0) -> (i64) {
    %actual = memref.load %out[%j] : memref<32xf64>
    %expectedValue = memref.load %expected[%j] : memref<16xf64>
    %equal = arith.cmpf oeq, %actual, %expectedValue : f64
    %error = arith.select %equal, %c0, %c1 : i64
    %errorsOut = arith.addi %errorsIn, %error : i64
    scf.yield %errorsOut : i64
  }

  memref.dealloc %in : memref<32xf64>
  memref.dealloc %comm : memref<32xf64>
  memref.dealloc %out : memref<32xf64>
  return %errors : i64
}
