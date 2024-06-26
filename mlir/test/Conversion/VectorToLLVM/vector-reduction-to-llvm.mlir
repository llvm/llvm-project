// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-vector-to-llvm='reassociate-fp-reductions' -split-input-file | FileCheck %s --check-prefix=REASSOC

// CHECK-LABEL: @reduce_add_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<16xf32>) -> f32
//      CHECK: return %[[V]] : f32
//
// REASSOC-LABEL: @reduce_add_f32(
// REASSOC-SAME: %[[A:.*]]: vector<16xf32>)
//      REASSOC: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      REASSOC: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// REASSOC-SAME: <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f32, vector<16xf32>) -> f32
//      REASSOC: return %[[V]] : f32
//
func.func @reduce_add_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<16xf32> into f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @reduce_add_f32_always_reassoc(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
/// Note: the reassoc flag remains even though the pass sets reassociate-fp-reduction to false.
/// Ponder whether this flag really is a property of the pass / pattern..
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f32, vector<16xf32>) -> f32
//      CHECK: return %[[V]] : f32
//
// REASSOC-LABEL: @reduce_add_f32_always_reassoc(
// REASSOC-SAME: %[[A:.*]]: vector<16xf32>)
//      REASSOC: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      REASSOC: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// REASSOC-SAME: <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f32, vector<16xf32>) -> f32
//      REASSOC: return %[[V]] : f32
//
func.func @reduce_add_f32_always_reassoc(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 fastmath<reassoc> : vector<16xf32> into f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @reduce_mul_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fmul"(%[[C]], %[[A]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<nnan, ninf>}> : (f32, vector<16xf32>) -> f32
//      CHECK: return %[[V]] : f32
//
// REASSOC-LABEL: @reduce_mul_f32(
// REASSOC-SAME: %[[A:.*]]: vector<16xf32>)
//      REASSOC: %[[C:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
//      REASSOC: %[[V:.*]] = "llvm.intr.vector.reduce.fmul"(%[[C]], %[[A]])
// REASSOC-SAME: <{fastmathFlags = #llvm.fastmath<nnan, ninf, reassoc>}> : (f32, vector<16xf32>) -> f32
//      REASSOC: return %[[V]] : f32
//
func.func @reduce_mul_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction <mul>, %arg0 fastmath<nnan, ninf> : vector<16xf32> into f32
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

func.func @masked_reduce_add_f32_scalable(%arg0: vector<[16]xf32>, %mask : vector<[16]xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <add>, %arg0 : vector<[16]xf32> into f32 } : vector<[16]xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_add_f32_scalable(
// CHECK-SAME:                              %[[INPUT:.*]]: vector<[16]xf32>,
// CHECK-SAME:                              %[[MASK:.*]]: vector<[16]xi1>) -> f32 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           %[[VL_BASE:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK:           %[[CAST_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK:           %[[CAST_I32:.*]] = arith.index_cast %[[CAST_IDX]] : index to i32
// CHECK:           %[[VL_MUL:.*]] = arith.muli %[[VL_BASE]], %[[CAST_I32]] : i32
// CHECK:           "llvm.intr.vp.reduce.fadd"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL_MUL]]) : (f32, vector<[16]xf32>, vector<[16]xi1>, i32) -> f32


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
  %0 = vector.mask %mask { vector.reduction <minnumf>, %arg0 : vector<16xf32> into f32 } : vector<16xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_minf_f32(
// CHECK-SAME:                                      %[[INPUT:.*]]: vector<16xf32>,
// CHECK-SAME:                                      %[[MASK:.*]]: vector<16xi1>) -> f32 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0xFFC00000 : f32) : f32
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.fmin"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (f32, vector<16xf32>, vector<16xi1>, i32) -> f32

// -----

func.func @masked_reduce_minf_f32_scalable(%arg0: vector<[16]xf32>, %mask : vector<[16]xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <minnumf>, %arg0 : vector<[16]xf32> into f32 } : vector<[16]xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_minf_f32_scalable(
// CHECK-SAME:                                      %[[INPUT:.*]]: vector<[16]xf32>,
// CHECK-SAME:                                      %[[MASK:.*]]: vector<[16]xi1>) -> f32 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0xFFC00000 : f32) : f32
// CHECK:           %[[VL_BASE:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK:           %[[CAST_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK:           %[[CAST_I32:.*]] = arith.index_cast %[[CAST_IDX]] : index to i32
// CHECK:           %[[VL_MUL:.*]] = arith.muli %[[VL_BASE]], %[[CAST_I32]] : i32
// CHECK:           "llvm.intr.vp.reduce.fmin"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL_MUL]]) : (f32, vector<[16]xf32>, vector<[16]xi1>, i32) -> f32

// -----

func.func @masked_reduce_maxf_f32(%arg0: vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <maxnumf>, %arg0 : vector<16xf32> into f32 } : vector<16xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_maxf_f32(
// CHECK-SAME:                                      %[[INPUT:.*]]: vector<16xf32>,
// CHECK-SAME:                                      %[[MASK:.*]]: vector<16xi1>) -> f32 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0x7FC00000 : f32) : f32
// CHECK:           %[[VL:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           "llvm.intr.vp.reduce.fmax"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL]]) : (f32, vector<16xf32>, vector<16xi1>, i32) -> f32

// -----

func.func @masked_reduce_maximumf_f32(%arg0: vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <maximumf>, %arg0 : vector<16xf32> into f32 } : vector<16xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_maximumf_f32(
// CHECK-SAME:                                      %[[INPUT:.*]]: vector<16xf32>,
// CHECK-SAME:                                      %[[MASK:.*]]: vector<16xi1>) -> f32 {
// CHECK:           %[[MASK_NEUTRAL:.*]] = llvm.mlir.constant(dense<-1.401300e-45> : vector<16xf32>) : vector<16xf32>
// CHECK:           %[[MASKED:.*]] = llvm.select %[[MASK]], %[[INPUT]], %[[MASK_NEUTRAL]] : vector<16xi1>, vector<16xf32>
// CHECK:           %[[RESULT:.*]] = llvm.intr.vector.reduce.fmaximum(%[[MASKED]])  : (vector<16xf32>) -> f32
// CHECK:           return %[[RESULT]]

// -----

func.func @masked_reduce_minimumf_f32(%arg0: vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <minimumf>, %arg0 : vector<16xf32> into f32 } : vector<16xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @masked_reduce_minimumf_f32(
// CHECK-SAME:                                      %[[INPUT:.*]]: vector<16xf32>,
// CHECK-SAME:                                      %[[MASK:.*]]: vector<16xi1>) -> f32 {
// CHECK:           %[[MASK_NEUTRAL:.*]] = llvm.mlir.constant(dense<3.40282347E+38> : vector<16xf32>) : vector<16xf32>
// CHECK:           %[[MASKED:.*]] = llvm.select %[[MASK]], %[[INPUT]], %[[MASK_NEUTRAL]] : vector<16xi1>, vector<16xf32>
// CHECK:           %[[RESULT:.*]] = llvm.intr.vector.reduce.fminimum(%[[MASKED]])  : (vector<16xf32>) -> f32
// CHECK:           return %[[RESULT]]

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

func.func @masked_reduce_add_i8_scalable(%arg0: vector<[32]xi8>, %mask : vector<[32]xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <add>, %arg0 : vector<[32]xi8> into i8 } : vector<[32]xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_add_i8_scalable(
// CHECK-SAME:                             %[[INPUT:.*]]: vector<[32]xi8>,
// CHECK-SAME:                             %[[MASK:.*]]: vector<[32]xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:           %[[VL_BASE:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK:           %[[CAST_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK:           %[[CAST_I32:.*]] = arith.index_cast %[[CAST_IDX]] : index to i32
// CHECK:           %[[VL_MUL:.*]] = arith.muli %[[VL_BASE]], %[[CAST_I32]] : i32
// CHECK:           "llvm.intr.vp.reduce.add"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL_MUL]]) : (i8, vector<[32]xi8>, vector<[32]xi1>, i32) -> i8


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

func.func @masked_reduce_minui_i8_scalable(%arg0: vector<[32]xi8>, %mask : vector<[32]xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <minui>, %arg0 : vector<[32]xi8> into i8 } : vector<[32]xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_minui_i8_scalable(
// CHECK-SAME:                               %[[INPUT:.*]]: vector<[32]xi8>,
// CHECK-SAME:                               %[[MASK:.*]]: vector<[32]xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(-1 : i8) : i8
// CHECK:           %[[VL_BASE:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK:           %[[CAST_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK:           %[[CAST_I32:.*]] = arith.index_cast %[[CAST_IDX]] : index to i32
// CHECK:           %[[VL_MUL:.*]] = arith.muli %[[VL_BASE]], %[[CAST_I32]] : i32
// CHECK:           "llvm.intr.vp.reduce.umin"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL_MUL]]) : (i8, vector<[32]xi8>, vector<[32]xi1>, i32) -> i8

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

func.func @masked_reduce_maxsi_i8_scalable(%arg0: vector<[32]xi8>, %mask : vector<[32]xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <maxsi>, %arg0 : vector<[32]xi8> into i8 } : vector<[32]xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_maxsi_i8_scalable(
// CHECK-SAME:                               %[[INPUT:.*]]: vector<[32]xi8>,
// CHECK-SAME:                               %[[MASK:.*]]: vector<[32]xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(-128 : i8) : i8
// CHECK:           %[[VL_BASE:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK:           %[[CAST_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK:           %[[CAST_I32:.*]] = arith.index_cast %[[CAST_IDX]] : index to i32
// CHECK:           %[[VL_MUL:.*]] = arith.muli %[[VL_BASE]], %[[CAST_I32]] : i32
// CHECK:           "llvm.intr.vp.reduce.smax"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL_MUL]]) : (i8, vector<[32]xi8>, vector<[32]xi1>, i32) -> i8

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

// -----

func.func @masked_reduce_xor_i8_scalable(%arg0: vector<[32]xi8>, %mask : vector<[32]xi1>) -> i8 {
  %0 = vector.mask %mask { vector.reduction <xor>, %arg0 : vector<[32]xi8> into i8 } : vector<[32]xi1> -> i8
  return %0 : i8
}

// CHECK-LABEL:   func.func @masked_reduce_xor_i8_scalable(
// CHECK-SAME:                             %[[INPUT:.*]]: vector<[32]xi8>,
// CHECK-SAME:                             %[[MASK:.*]]: vector<[32]xi1>) -> i8 {
// CHECK:           %[[NEUTRAL:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:           %[[VL_BASE:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK:           %[[CAST_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK:           %[[CAST_I32:.*]] = arith.index_cast %[[CAST_IDX]] : index to i32
// CHECK:           %[[VL_MUL:.*]] = arith.muli %[[VL_BASE]], %[[CAST_I32]] : i32
// CHECK:           "llvm.intr.vp.reduce.xor"(%[[NEUTRAL]], %[[INPUT]], %[[MASK]], %[[VL_MUL]]) : (i8, vector<[32]xi8>, vector<[32]xi1>, i32) -> i8


