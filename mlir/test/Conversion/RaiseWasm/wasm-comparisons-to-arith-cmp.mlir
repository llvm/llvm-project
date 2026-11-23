// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s


module {
  wasmssa.func @func_lt_si32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.lt_si %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }

  wasmssa.func @func_le_si32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.le_si %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_lt_ui32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.lt_ui %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_le_ui32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.le_ui %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_gt_si32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.gt_si %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_gt_ui32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.gt_ui %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ge_si32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.ge_si %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ge_ui32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.ge_ui %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_lt_si64() -> i32 {
    %0 = wasmssa.const 12 : i64
    %1 = wasmssa.const 50 : i64
    %2 = wasmssa.lt_si %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_le_si64() -> i32 {
    %0 = wasmssa.const 12 : i64
    %1 = wasmssa.const 50 : i64
    %2 = wasmssa.le_si %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_lt_ui_i64() -> i32 {
    %0 = wasmssa.const 12 : i64
    %1 = wasmssa.const 50 : i64
    %2 = wasmssa.lt_ui %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_le_ui_i64() -> i32 {
    %0 = wasmssa.const 12 : i64
    %1 = wasmssa.const 50 : i64
    %2 = wasmssa.le_ui %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_gt_si_i64() -> i32 {
    %0 = wasmssa.const 12 : i64
    %1 = wasmssa.const 50 : i64
    %2 = wasmssa.gt_si %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_gt_ui_i64() -> i32 {
    %0 = wasmssa.const 12 : i64
    %1 = wasmssa.const 50 : i64
    %2 = wasmssa.gt_ui %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ge_si_i64() -> i32 {
    %0 = wasmssa.const 12 : i64
    %1 = wasmssa.const 50 : i64
    %2 = wasmssa.ge_si %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ge_ui_i64() -> i32 {
    %0 = wasmssa.const 12 : i64
    %1 = wasmssa.const 50 : i64
    %2 = wasmssa.ge_ui %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_lt_f32() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f32
    %1 = wasmssa.const 1.400000e+01 : f32
    %2 = wasmssa.lt %0 %1 : f32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_le_f32() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f32
    %1 = wasmssa.const 1.400000e+01 : f32
    %2 = wasmssa.le %0 %1 : f32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_gt_f32() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f32
    %1 = wasmssa.const 1.400000e+01 : f32
    %2 = wasmssa.gt %0 %1 : f32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ge_f32() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f32
    %1 = wasmssa.const 1.400000e+01 : f32
    %2 = wasmssa.ge %0 %1 : f32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_lt_f64() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f64
    %1 = wasmssa.const 1.400000e+01 : f64
    %2 = wasmssa.lt %0 %1 : f64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_le_f64() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f64
    %1 = wasmssa.const 1.400000e+01 : f64
    %2 = wasmssa.le %0 %1 : f64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_gt_f64() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f64
    %1 = wasmssa.const 1.400000e+01 : f64
    %2 = wasmssa.gt %0 %1 : f64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ge_f64() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f64
    %1 = wasmssa.const 1.400000e+01 : f64
    %2 = wasmssa.ge %0 %1 : f64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_eq_i32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.eq %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_eq_i64() -> i32 {
    %0 = wasmssa.const 20 : i64
    %1 = wasmssa.const 5 : i64
    %2 = wasmssa.eq %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_eq_f32() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f32
    %1 = wasmssa.const 1.400000e+01 : f32
    %2 = wasmssa.eq %0 %1 : f32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_eq_f64() -> i32 {
    %0 = wasmssa.const 1.700000e+01 : f64
    %1 = wasmssa.const 0.000000e+00 : f64
    %2 = wasmssa.eq %0 %1 : f64 -> i32
    wasmssa.return %2 : i32
  }
    wasmssa.func @func_ne_i32() -> i32 {
    %0 = wasmssa.const 12 : i32
    %1 = wasmssa.const 50 : i32
    %2 = wasmssa.ne %0 %1 : i32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ne_i64() -> i32 {
    %0 = wasmssa.const 20 : i64
    %1 = wasmssa.const 5 : i64
    %2 = wasmssa.ne %0 %1 : i64 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ne_f32() -> i32 {
    %0 = wasmssa.const 5.000000e+00 : f32
    %1 = wasmssa.const 1.400000e+01 : f32
    %2 = wasmssa.ne %0 %1 : f32 -> i32
    wasmssa.return %2 : i32
  }
  wasmssa.func @func_ne_f64() -> i32 {
    %0 = wasmssa.const 1.700000e+01 : f64
    %1 = wasmssa.const 0.000000e+00 : f64
    %2 = wasmssa.ne %0 %1 : f64 -> i32
    wasmssa.return %2 : i32
  }
}
// CHECK-LABEL:   func.func @func_lt_si32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_le_si32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi sle, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_lt_ui32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ult, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_le_ui32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ule, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_gt_si32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_gt_ui32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ugt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ge_si32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi sge, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ge_ui32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi uge, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_lt_si64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_le_si64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi sle, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_lt_ui_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ult, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_le_ui_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ule, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_gt_si_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_gt_ui_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ugt, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ge_si_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi sge, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ge_ui_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi uge, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_lt_f32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_le_f32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = arith.cmpf ole, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_gt_f32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = arith.cmpf ogt, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ge_f32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = arith.cmpf oge, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_lt_f64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f64
// CHECK:           %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_le_f64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f64
// CHECK:           %[[VAL_2:.*]] = arith.cmpf ole, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_gt_f64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f64
// CHECK:           %[[VAL_2:.*]] = arith.cmpf ogt, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ge_f64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f64
// CHECK:           %[[VAL_2:.*]] = arith.cmpf oge, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_eq_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_eq_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 20 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 5 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_eq_f32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_eq_f64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1.700000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ne_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 50 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ne_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 20 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 5 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ne_f32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = arith.cmpf one, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @func_ne_f64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1.700000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = arith.cmpf one, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32
