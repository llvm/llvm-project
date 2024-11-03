// RUN: mlir-opt %s -convert-vector-to-llvm='use-opaque-pointers=1' -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-vector-to-llvm='reassociate-fp-reductions use-opaque-pointers=1' -split-input-file | FileCheck %s --check-prefix=REASSOC

// CHECK-LABEL: @reduce_add_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: {reassoc = false} : (f32, vector<16xf32>) -> f32
//      CHECK: return %[[V]] : f32
//
// REASSOC-LABEL: @reduce_add_f32(
// REASSOC-SAME: %[[A:.*]]: vector<16xf32>)
//      REASSOC: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      REASSOC: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// REASSOC-SAME: {reassoc = true} : (f32, vector<16xf32>) -> f32
//      REASSOC: return %[[V]] : f32
//
func.func @reduce_add_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<16xf32> into f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @reduce_mul_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fmul"(%[[C]], %[[A]])
// CHECK-SAME: {reassoc = false} : (f32, vector<16xf32>) -> f32
//      CHECK: return %[[V]] : f32
//
// REASSOC-LABEL: @reduce_mul_f32(
// REASSOC-SAME: %[[A:.*]]: vector<16xf32>)
//      REASSOC: %[[C:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
//      REASSOC: %[[V:.*]] = "llvm.intr.vector.reduce.fmul"(%[[C]], %[[A]])
// REASSOC-SAME: {reassoc = true} : (f32, vector<16xf32>) -> f32
//      REASSOC: return %[[V]] : f32
//
func.func @reduce_mul_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction <mul>, %arg0 : vector<16xf32> into f32
  return %0 : f32
}

// -----

func.func @masked_reduce_add_f32(%arg0: vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <add>, %arg0 : vector<16xf32> into f32 } : vector<16xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_add_f32(
// CHECK-SAME:                              %[[INPUT:.*]]: vector<16xf32>,
// CHECK-SAME:                              %[[MASK:.*]]: vector<16xi1>) -> f32 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.fadd"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (f32, vector<16xf32>, vector<16xi1>, i32) -> f32


// -----

func.func @masked_reduce_mul_f32(%arg0: vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <mul>, %arg0 : vector<16xf32> into f32 } : vector<16xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_mul_f32(
// CHECK-SAME:                              %[[INPUT:.*]]: vector<16xf32>,
// CHECK-SAME:                              %[[MASK:.*]]: vector<16xi1>) -> f32 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.fmul"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (f32, vector<16xf32>, vector<16xi1>, i32) -> f32


// -----

func.func @masked_reduce_minf_f32(%arg0: vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <minf>, %arg0 : vector<16xf32> into f32 } : vector<16xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_minf_f32(
// CHECK-SAME:                                      %[[INPUT:.*]]: vector<16xf32>,
// CHECK-SAME:                                      %[[MASK:.*]]: vector<16xi1>) -> f32 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0xFFC00000 : f32) : f32
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.fmin"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (f32, vector<16xf32>, vector<16xi1>, i32) -> f32

// -----

func.func @masked_reduce_maxf_f32(%arg0: vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <maxf>, %arg0 : vector<16xf32> into f32 } : vector<16xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_maxf_f32(
// CHECK-SAME:                                      %[[INPUT:.*]]: vector<16xf32>,
// CHECK-SAME:                                      %[[MASK:.*]]: vector<16xi1>) -> f32 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0x7FC00000 : f32) : f32
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.fmax"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (f32, vector<16xf32>, vector<16xi1>, i32) -> f32

// -----

func.func @masked_reduce_add_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <add>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_add_i8(
// CHECK-SAME:                             %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                             %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.add"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8


// -----

func.func @masked_reduce_mul_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <mul>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_mul_i8(
// CHECK-SAME:                             %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                             %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = "llvm.intr.vp.reduce.mul"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8

// -----

func.func @masked_reduce_minui_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <minui>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_minui_i8(
// CHECK-SAME:                               %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                               %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(-1 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.umin"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8

// -----

func.func @masked_reduce_maxui_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <maxui>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_maxui_i8(
// CHECK-SAME:                               %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                               %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.umax"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8

// -----

func.func @masked_reduce_minsi_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <minsi>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_minsi_i8(
// CHECK-SAME:                               %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                               %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(127 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.smin"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8

// -----

func.func @masked_reduce_maxsi_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <maxsi>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_maxsi_i8(
// CHECK-SAME:                               %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                               %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(-128 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.smax"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8

// -----

func.func @masked_reduce_or_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <or>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_or_i8(
// CHECK-SAME:                            %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                            %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.or"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8


// -----

func.func @masked_reduce_and_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <and>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_and_i8(
// CHECK-SAME:                             %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                             %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(-1 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.and"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8

// -----

func.func @masked_reduce_xor_i8(%arg0: vector<32xi8>, %mask : vector<32xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <xor>, %arg0 : vector<32xi8> into i8 } : vector<32xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_xor_i8(
// CHECK-SAME:                             %[[INPUT:.*]]: vector<32xi8>,
// CHECK-SAME:                             %[[MASK:.*]]: vector<32xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.xor"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (i8, vector<32xi8>, vector<32xi1>, i32) -> i8


