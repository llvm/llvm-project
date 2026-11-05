// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s
// RUN: mlir-opt %s -arith-int-range-narrowing='int-bitwidths-supported=8,16,32' -int-range-optimizations | FileCheck %s

// CHECK-LABEL: @select_poison
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]]
func.func @select_poison(%c: i1, %x: i32) -> i1 {
  %p = ub.poison : i32
  %zero = arith.constant 0 : i32
  %seven = arith.constant 7 : i32
  %lo = arith.maxsi %x, %zero : i32
  %bounded = arith.minsi %lo, %seven : i32
  %v = arith.select %c, %bounded, %p : i32
  %r = arith.cmpi sge, %v, %zero : i32
  return %r : i1
}

// CHECK-LABEL: @select_unknown
// CHECK: arith.select
// CHECK: %[[CMP:.*]] = arith.cmpi
// CHECK: return %[[CMP]]
func.func @select_unknown(%c: i1, %x: i32, %y: i32) -> i1 {
  %zero = arith.constant 0 : i32
  %lo = arith.maxsi %x, %zero : i32
  %v = arith.select %c, %lo, %y : i32
  %r = arith.cmpi sge, %v, %zero : i32
  return %r : i1
}

// CHECK-LABEL: @poison_condition
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: return %[[P]]
func.func @poison_condition(%x: i32, %y: i32) -> i32 {
  %p = ub.poison : i1
  %v = arith.select %p, %x, %y : i32
  return %v : i32
}

// CHECK-LABEL: @overflow_divisor
// CHECK: %[[P:.*]] = index.constant -1
// CHECK: return %[[P]]
func.func @overflow_divisor() -> index {
  %z = arith.constant 0 : index
  %o = arith.constant 1 : index
  %p = arith.subi %z, %o overflow<nsw, nuw> : index
  %q = index.divs %o, %p
  return %q : index
}

// CHECK-LABEL: @vector_insert
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]]
func.func @vector_insert(%x: i32) -> i1 {
  %zero = arith.constant 0 : i32
  %lo = arith.maxsi %x, %zero : i32
  %p = ub.poison : vector<2xi32>
  %v = vector.insert %lo, %p[0] : i32 into vector<2xi32>
  %w = vector.insert %lo, %v[1] : i32 into vector<2xi32>
  %e = vector.extract %w[0] : i32 from vector<2xi32>
  %r = arith.cmpi sge, %e, %zero : i32
  return %r : i1
}

// CHECK-LABEL: @cfg_merge
// CHECK: %[[SEVEN:.*]] = arith.constant 7 : i32
// CHECK: return %[[SEVEN]]
func.func @cfg_merge(%c: i1) -> i32 {
  %p = ub.poison : i32
  %seven = arith.constant 7 : i32
  cf.cond_br %c, ^left, ^right
^left:
  cf.br ^merge(%p : i32)
^right:
  cf.br ^merge(%seven : i32)
^merge(%v: i32):
  return %v : i32
}

// CHECK-LABEL: @freeze
// CHECK: llvm.freeze
// CHECK: %[[CMP:.*]] = arith.cmpi
// CHECK: return %[[CMP]]
func.func @freeze(%c: i1, %x: i32) -> i1 {
  %p = ub.poison : i32
  %zero = arith.constant 0 : i32
  %lo = arith.maxsi %x, %zero : i32
  %v = arith.select %c, %lo, %p : i32
  %f = llvm.freeze %v : i32
  %r = arith.cmpi sge, %f, %zero : i32
  return %r : i1
}

// CHECK-LABEL: @non_integer_ready
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: return %[[ONE]]
func.func @non_integer_ready(%tensor: tensor<3x5xf32>, %c: i1) -> index {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %d = tensor.dim %tensor, %zero : tensor<3x5xf32>
  %r = arith.divui %d, %d : index
  return %r : index
}

// CHECK-LABEL: @loop_carried_poison
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]]
func.func @loop_carried_poison(%n: index, %c: i1, %x: i32) -> i1 {
  %z = arith.constant 0 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %lo = arith.maxsi %x, %zero : i32
  %p = ub.poison : i32
  %v = scf.for %i = %z to %n step %step iter_args(%a = %p) -> i32 {
    %next = arith.select %c, %lo, %a : i32
    scf.yield %next : i32
  }
  %r = arith.cmpi sge, %v, %zero : i32
  return %r : i1
}

// CHECK-LABEL: @non_integer_cfg
// CHECK: cf.cond_br
// CHECK: return
func.func @non_integer_cfg(%x: f32) -> f32 {
  %zero = arith.constant 0.0 : f32
  %c = arith.cmpf olt, %x, %zero : f32
  cf.cond_br %c, ^left, ^right
^left:
  %n = arith.negf %x : f32
  cf.br ^merge(%n : f32)
^right:
  cf.br ^merge(%x : f32)
^merge(%v: f32):
  return %v : f32
}

// CHECK-LABEL: @nvvm_half_open
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]]
func.func @nvvm_half_open() -> i1 {
  %v = nvvm.read.ptx.sreg.tid.x range <i32, 0, 32> : i32
  %limit = arith.constant 32 : i32
  %c = arith.cmpi ult, %v, %limit : i32
  return %c : i1
}

// CHECK-LABEL: @nvvm_wrapped
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]]
func.func @nvvm_wrapped() -> i1 {
  %v = nvvm.read.ptx.sreg.tid.x range <i32, -4, 4> : i32
  %limit = arith.constant 4 : i32
  %c = arith.cmpi slt, %v, %limit : i32
  return %c : i1
}

// CHECK-LABEL: @nvvm_full
// CHECK: nvvm.read.ptx.sreg
// CHECK: %[[CMP:.*]] = arith.cmpi
// CHECK: return %[[CMP]]
func.func @nvvm_full() -> i1 {
  %v = nvvm.read.ptx.sreg.tid.x range <i32, -1, -1> : i32
  %zero = arith.constant 0 : i32
  %c = arith.cmpi eq, %v, %zero : i32
  return %c : i1
}

// CHECK-LABEL: @resource_integer_constant
// CHECK: arith.constant dense_resource
// CHECK: %[[CMP:.*]] = arith.cmpi
// CHECK: return %[[CMP]]
func.func @resource_integer_constant(%x: tensor<i8>) -> tensor<i1> {
  %resource = arith.constant dense_resource<range_test> : tensor<i8>
  %sum = arith.addi %resource, %x : tensor<i8>
  %zero = arith.constant dense<0> : tensor<i8>
  %c = arith.cmpi eq, %sum, %zero : tensor<i8>
  return %c : tensor<i1>
}

// CHECK-LABEL: @undef_select
// CHECK: llvm.mlir.undef
// CHECK: arith.select
// CHECK: %[[CMP:.*]] = arith.cmpi
// CHECK: return %[[CMP]]
func.func @undef_select(%cond: i1) -> i1 {
  %u = llvm.mlir.undef : i32
  %zero = arith.constant 0 : i32
  %v = arith.select %cond, %zero, %u : i32
  %c = arith.cmpi eq, %v, %zero : i32
  return %c : i1
}

// CHECK-LABEL: @float_vector_extract
// CHECK: %[[R:.*]] = vector.extract
// CHECK: return %[[R]]
func.func @float_vector_extract(%v: vector<2xf32>, %i: index) -> f32 {
  %r = vector.extract %v[%i] : f32 from vector<2xf32>
  return %r : f32
}

// CHECK-LABEL: @integer_memref_has_no_range
// CHECK: %[[THREE:.*]] = arith.constant 3 : index
// CHECK: return %[[THREE]]
func.func @integer_memref_has_no_range(%m: memref<3xindex>) -> index {
  %zero = arith.constant 0 : index
  %cast = arith.index_cast %m : memref<3xindex> to memref<3xi32>
  %d = memref.dim %cast, %zero : memref<3xi32>
  return %d : index
}

// CHECK-LABEL: @float_tensor_select
// CHECK: %[[THREE:.*]] = arith.constant 3 : index
// CHECK: return %[[THREE]]
func.func @float_tensor_select(%c: i1, %a: tensor<3x5xf32>,
                               %b: tensor<3x5xf32>) -> index {
  %zero = arith.constant 0 : index
  %t = arith.select %c, %a, %b : tensor<3x5xf32>
  %d = tensor.dim %t, %zero : tensor<3x5xf32>
  return %d : index
}
