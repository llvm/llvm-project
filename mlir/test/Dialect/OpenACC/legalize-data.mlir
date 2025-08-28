// RUN: mlir-opt -split-input-file --openacc-legalize-data-values %s | FileCheck %s --check-prefixes=CHECK,DEVICE
// RUN: mlir-opt -split-input-file --openacc-legalize-data-values=host-to-device=false %s | FileCheck %s --check-prefixes=CHECK,HOST

func.func @test(%a: memref<10xf32>, %i : index) {
  %create = acc.create varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.parallel dataOperands(%create : memref<10xf32>) {
    %ci = memref.load %a[%i] : memref<10xf32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>, %[[I:.*]]: index)
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.parallel dataOperands(%[[CREATE]] : memref<10xf32>) {
// DEVICE:   %{{.*}} = memref.load %[[CREATE]][%[[I]]] : memref<10xf32>
// HOST:    %{{.*}} = memref.load %[[A]][%[[I]]] : memref<10xf32>
// CHECK:   acc.yield
// CHECK: }

// -----

func.func @test(%a: memref<10xf32>, %i : index) {
  %create = acc.create varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.serial dataOperands(%create : memref<10xf32>) {
    %ci = memref.load %a[%i] : memref<10xf32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>, %[[I:.*]]: index)
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.serial dataOperands(%[[CREATE]] : memref<10xf32>) {
// DEVICE:   %{{.*}} = memref.load %[[CREATE]][%[[I]]] : memref<10xf32>
// HOST:    %{{.*}} = memref.load %[[A]][%[[I]]] : memref<10xf32>
// CHECK:   acc.yield
// CHECK: }

// -----

func.func @test(%a: memref<10xf32>, %i : index) {
  %create = acc.create varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.kernels dataOperands(%create : memref<10xf32>) {
    %ci = memref.load %a[%i] : memref<10xf32>
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>, %[[I:.*]]: index)
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.kernels dataOperands(%[[CREATE]] : memref<10xf32>) {
// DEVICE:   %{{.*}} = memref.load %[[CREATE]][%[[I]]] : memref<10xf32>
// HOST:    %{{.*}} = memref.load %[[A]][%[[I]]] : memref<10xf32>
// CHECK:   acc.terminator
// CHECK: }

// -----

func.func @test(%a: memref<10xf32>, %i : index) {
  %create = acc.create varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.data dataOperands(%create : memref<10xf32>) {
    %c0 = arith.constant 0.000000e+00 : f32
    memref.store %c0, %a[%i] : memref<10xf32>
    acc.serial {
      %cs = memref.load %a[%i] : memref<10xf32>
      acc.yield
    }
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>, %[[I:.*]]: index)
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.data dataOperands(%[[CREATE]] : memref<10xf32>) {
// CHECK: memref.store %{{.*}}, %[[A]][%[[I]]] : memref<10xf32>
// DEVICE:   %{{.*}} = memref.load %[[CREATE]][%[[I]]] : memref<10xf32>
// HOST:    %{{.*}} = memref.load %[[A]][%[[I]]] : memref<10xf32>
// CHECK:   acc.terminator
// CHECK: }

// -----

func.func @test(%a: memref<10xf32>) {
  %lb = arith.constant 0 : index
  %st = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %create = acc.create varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.parallel dataOperands(%create : memref<10xf32>) {
    acc.loop control(%i : index) = (%lb : index) to (%c10 : index) step (%st : index) {
      %ci = memref.load %a[%i] : memref<10xf32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>)
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.parallel dataOperands(%[[CREATE]] : memref<10xf32>) {
// CHECK:   acc.loop control(%[[I:.*]] : index) = (%{{.*}} : index) to (%{{.*}} : index)  step (%{{.*}} : index) {
// DEVICE:    %{{.*}} = memref.load %[[CREATE:.*]][%[[I]]] : memref<10xf32>
// CHECK:     acc.yield
// CHECK:   } attributes {independent = [#acc.device_type<none>]}
// CHECK:   acc.yield
// CHECK: }

// -----

acc.private.recipe @privatization_memref_10_f32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

func.func @test(%a: memref<10xf32>) {
  %lb = arith.constant 0 : index
  %st = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %p1 = acc.private varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.parallel private(@privatization_memref_10_f32 -> %p1 : memref<10xf32>) {
    acc.loop control(%i : index) = (%lb : index) to (%c10 : index) step (%st : index) {
      %ci = memref.load %a[%i] : memref<10xf32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>)
// CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.parallel private(@privatization_memref_10_f32 -> %[[PRIVATE]] : memref<10xf32>) {
// CHECK:   acc.loop control(%[[I:.*]] : index) = (%{{.*}} : index) to (%{{.*}} : index)  step (%{{.*}} : index) {
// DEVICE:    %{{.*}} = memref.load %[[PRIVATE:.*]][%[[I]]] : memref<10xf32>
// CHECK:     acc.yield
// CHECK:   } attributes {independent = [#acc.device_type<none>]}
// CHECK:   acc.yield
// CHECK: }

// -----

acc.private.recipe @privatization_memref_10_f32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

func.func @test(%a: memref<10xf32>) {
  %lb = arith.constant 0 : index
  %st = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %p1 = acc.private varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.parallel {
    acc.loop private(@privatization_memref_10_f32 -> %p1 : memref<10xf32>) control(%i : index) = (%lb : index) to (%c10 : index) step (%st : index) {
      %ci = memref.load %a[%i] : memref<10xf32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>)
// CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.parallel  {
// CHECK:   acc.loop private(@privatization_memref_10_f32 -> %[[PRIVATE]] : memref<10xf32>) control(%[[I:.*]] : index) = (%{{.*}} : index) to (%{{.*}} : index)  step (%{{.*}} : index) {
// DEVICE:    %{{.*}} = memref.load %[[PRIVATE:.*]][%[[I]]] : memref<10xf32>
// CHECK:     acc.yield
// CHECK:   } attributes {independent = [#acc.device_type<none>]}
// CHECK:   acc.yield
// CHECK: }

// -----

acc.private.recipe @privatization_memref_10_f32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

func.func @test(%a: memref<10xf32>) {
  %lb = arith.constant 0 : index
  %st = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %p1 = acc.private varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.serial private(@privatization_memref_10_f32 -> %p1 : memref<10xf32>) {
    acc.loop control(%i : index) = (%lb : index) to (%c10 : index) step (%st : index) {
      %ci = memref.load %a[%i] : memref<10xf32>
      acc.yield
    } attributes {seq = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>)
// CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.serial private(@privatization_memref_10_f32 -> %[[PRIVATE]] : memref<10xf32>) {
// CHECK:   acc.loop control(%[[I:.*]] : index) = (%{{.*}} : index) to (%{{.*}} : index)  step (%{{.*}} : index) {
// DEVICE:    %{{.*}} = memref.load %[[PRIVATE:.*]][%[[I]]] : memref<10xf32>
// CHECK:     acc.yield
// CHECK:   } attributes {seq = [#acc.device_type<none>]}
// CHECK:   acc.yield
// CHECK: }

// -----

func.func @test(%a: memref<10xf32>) {
  %devptr = acc.use_device varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
  acc.host_data dataOperands(%devptr : memref<10xf32>) {
    func.call @foo(%a) : (memref<10xf32>) -> ()
    acc.terminator
  }
  return
}
func.func private @foo(memref<10xf32>)

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>)
// CHECK: %[[USE_DEVICE:.*]] = acc.use_device varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
// CHECK: acc.host_data dataOperands(%[[USE_DEVICE]] : memref<10xf32>) {
// DEVICE:   func.call @foo(%[[USE_DEVICE]]) : (memref<10xf32>) -> ()
// CHECK:   acc.terminator
// CHECK: }

// -----

func.func @test(%a: memref<10xf32>) {
  %declare = acc.create varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32> {name = "arr"}
  %token = acc.declare_enter dataOperands(%declare : memref<10xf32>)
  acc.kernels dataOperands(%declare : memref<10xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1.000000e+00 : f32
    memref.store %c1, %a[%c0] : memref<10xf32>
    acc.terminator
  }
  acc.declare_exit token(%token) dataOperands(%declare : memref<10xf32>)
  return
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[A:.*]]: memref<10xf32>)
// CHECK: %[[DECLARE:.*]] = acc.create varPtr(%[[A]] : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32> {name = "arr"}
// CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[DECLARE]] : memref<10xf32>)
// CHECK: acc.kernels dataOperands(%[[DECLARE]] : memref<10xf32>) {
// DEVICE:   memref.store %{{.*}}, %[[DECLARE]][%{{.*}}] : memref<10xf32>
// HOST:    memref.store %{{.*}}, %[[A]][%{{.*}}] : memref<10xf32>
// CHECK:   acc.terminator
// CHECK: }
// CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[DECLARE]] : memref<10xf32>)
// CHECK: return