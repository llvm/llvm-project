// RUN: mlir-opt %s -convert-gpu-to-agxir='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 4 --wg 4 %t.mlir f64:1,0.1,1e308,3 f64:3,0.2,10,-0.0 f64*4 i32*4 | FileCheck %s

// Double precision arithmetic is computed in software: a / b + sqrt(a),
// with large values, a division by -0, and comparisons.
// CHECK: 2: 1.33333 0.816228 1e+307 -inf
// CHECK: 3: 0 0 1 1

gpu.module @kernels {
  gpu.func @f64(%a: memref<4xf64, 1>, %b: memref<4xf64, 1>, %out: memref<4xf64, 1>,
                %cmp: memref<4xi32, 1>) kernel {
    %i = gpu.thread_id x
    %x = memref.load %a[%i] : memref<4xf64, 1>
    %y = memref.load %b[%i] : memref<4xf64, 1>
    %q = arith.divf %x, %y : f64
    %s = math.sqrt %x : f64
    %r = arith.addf %q, %s : f64
    memref.store %r, %out[%i] : memref<4xf64, 1>
    %gt = arith.cmpf ogt, %x, %y : f64
    %z = arith.extui %gt : i1 to i32
    memref.store %z, %cmp[%i] : memref<4xi32, 1>
    gpu.return
  }
}
