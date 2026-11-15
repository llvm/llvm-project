// RUN: mlir-opt %s -acc-recipe-materialization | FileCheck %s

// A private recipe whose init region reads the original variable needs that
// variable on the device, so the materialization has to map its initial value.
acc.private.recipe @privatization_memref_dyn : memref<?xi32> init {
^bb0(%arg0: memref<?xi32>):
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi32>
  %0 = memref.alloca(%dim) : memref<?xi32>
  acc.yield %0 : memref<?xi32>
} destroy {
^bb0(%arg0: memref<?xi32>, %arg1: memref<?xi32>):
  acc.terminator
}

// The destroy region receives the original variable as well, so reading it
// there has the same effect.
acc.private.recipe @privatization_memref_destroy_reads : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
} destroy {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  acc.terminator
}

// A private recipe that ignores the original variable must not map anything.
acc.private.recipe @privatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
}

// CHECK-LABEL: func.func @private_dyn
// CHECK: %[[MAP:.*]] = acc.firstprivate_map varPtr(%{{.*}} : memref<?xi32>)
// CHECK: acc.parallel
// CHECK: %[[DIM:.*]] = memref.dim %[[MAP]]
// CHECK: %[[ALLOCA:.*]] = memref.alloca(%[[DIM]])
// CHECK: memref.store %{{.*}}, %[[ALLOCA]]

func.func @private_dyn(%arg0 : memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1336 = arith.constant 1336 : i32
  %priv = acc.private varPtr(%arg0 : memref<?xi32>) recipe(@privatization_memref_dyn) implicit(true) name("t") -> memref<?xi32>
  acc.parallel private(%priv : memref<?xi32>) {
    memref.store %c1336, %priv[%c0] : memref<?xi32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @private_destroy_reads
// CHECK: %[[MAP:.*]] = acc.firstprivate_map varPtr(%{{.*}} : memref<i32>)
// CHECK: acc.parallel
// CHECK: %[[LOAD:.*]] = memref.load %[[MAP]][]

func.func @private_destroy_reads(%arg0 : memref<i32>) {
  %c1336 = arith.constant 1336 : i32
  %priv = acc.private varPtr(%arg0 : memref<i32>) recipe(@privatization_memref_destroy_reads) implicit(true) name("t") -> memref<i32>
  acc.parallel private(%priv : memref<i32>) {
    memref.store %c1336, %priv[] : memref<i32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @private_scalar
// CHECK-NOT: acc.firstprivate_map
// CHECK: acc.parallel

func.func @private_scalar(%arg0 : memref<i32>) {
  %c1336 = arith.constant 1336 : i32
  %priv = acc.private varPtr(%arg0 : memref<i32>) recipe(@privatization_memref_i32) implicit(true) name("t") -> memref<i32>
  acc.parallel private(%priv : memref<i32>) {
    memref.store %c1336, %priv[] : memref<i32>
    acc.yield
  }
  return
}
