// RUN: yaml2obj %S/inputs/const.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s
/* Source code used to create this test:
(module
    (func(result i32)
        i32.const 1
    )
    (func (result i64)
        i64.const 3
    )
    (func (result f32)
        f32.const 4.0
    )
    (func (result f64)
        f64.const 9.0
    )
)
*/

// CHECK-LABEL:   wasmssa.func nested @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func nested @func_1() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 3 : i64
// CHECK:           wasmssa.return %[[VAL_0]] : i64
// CHECK:         }

// CHECK-LABEL:   wasmssa.func nested @func_2() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 4.000000e+00 : f32
// CHECK:           wasmssa.return %[[VAL_0]] : f32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func nested @func_3() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 9.000000e+00 : f64
// CHECK:           wasmssa.return %[[VAL_0]] : f64
// CHECK:         }
