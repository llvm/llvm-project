// RUN: yaml2obj %S/inputs/trunc_sat.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "trunc_sat_f32_s") (result i32)
        f32.const 12.1
        i32.trunc_sat_f32_s
    )

    (func (export "trunc_sat_f32_u") (result i32)
        f32.const 12.1
        i32.trunc_sat_f32_u
    )

    (func (export "trunc_sat_f64_s") (result i32)
        f64.const 12.1
        i32.trunc_sat_f64_s
    )

    (func (export "trunc_sat_f64_u") (result i32)
        f64.const 12.1
        i32.trunc_sat_f64_u
    )

    (func (export "i64_trunc_sat_f32_s") (result i64)
        f32.const 12.1
        i64.trunc_sat_f32_s
    )

    (func (export "i64_trunc_sat_f32_u") (result i64)
        f32.const 12.1
        i64.trunc_sat_f32_u
    )

    (func (export "i64_trunc_sat_f64_s") (result i64)
        f64.const 12.1
        i64.trunc_sat_f64_s
    )

    (func (export "i64_trunc_sat_f64_u") (result i64)
        f64.const 12.1
        i64.trunc_sat_f64_u
    )
)
*/


// CHECK-LABEL:   wasmssa.func exported @trunc_sat_f32_s() -> i32 {
// CHECK:           %[[CONST_0:.*]] = wasmssa.const 1.210000e+01 : f32
// CHECK:           %[[TRUNC_SAT_SI_0:.*]] = wasmssa.trunc_sat_si %[[CONST_0]] : f32 to i32
// CHECK:           wasmssa.return %[[TRUNC_SAT_SI_0]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func exported @trunc_sat_f32_u() -> i32 {
// CHECK:           %[[CONST_0:.*]] = wasmssa.const 1.210000e+01 : f32
// CHECK:           %[[TRUNC_SAT_UI_0:.*]] = wasmssa.trunc_sat_ui %[[CONST_0]] : f32 to i32
// CHECK:           wasmssa.return %[[TRUNC_SAT_UI_0]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func exported @trunc_sat_f64_s() -> i32 {
// CHECK:           %[[CONST_0:.*]] = wasmssa.const 1.210000e+01 : f64
// CHECK:           %[[TRUNC_SAT_SI_0:.*]] = wasmssa.trunc_sat_si %[[CONST_0]] : f64 to i32
// CHECK:           wasmssa.return %[[TRUNC_SAT_SI_0]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func exported @trunc_sat_f64_u() -> i32 {
// CHECK:           %[[CONST_0:.*]] = wasmssa.const 1.210000e+01 : f64
// CHECK:           %[[TRUNC_SAT_UI_0:.*]] = wasmssa.trunc_sat_ui %[[CONST_0]] : f64 to i32
// CHECK:           wasmssa.return %[[TRUNC_SAT_UI_0]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func exported @i64_trunc_sat_f32_s() -> i64 {
// CHECK:           %[[CONST_0:.*]] = wasmssa.const 1.210000e+01 : f32
// CHECK:           %[[TRUNC_SAT_SI_0:.*]] = wasmssa.trunc_sat_si %[[CONST_0]] : f32 to i64
// CHECK:           wasmssa.return %[[TRUNC_SAT_SI_0]] : i64
// CHECK:         }

// CHECK-LABEL:   wasmssa.func exported @i64_trunc_sat_f32_u() -> i64 {
// CHECK:           %[[CONST_0:.*]] = wasmssa.const 1.210000e+01 : f32
// CHECK:           %[[TRUNC_SAT_UI_0:.*]] = wasmssa.trunc_sat_ui %[[CONST_0]] : f32 to i64
// CHECK:           wasmssa.return %[[TRUNC_SAT_UI_0]] : i64
// CHECK:         }

// CHECK-LABEL:   wasmssa.func exported @i64_trunc_sat_f64_s() -> i64 {
// CHECK:           %[[CONST_0:.*]] = wasmssa.const 1.210000e+01 : f64
// CHECK:           %[[TRUNC_SAT_SI_0:.*]] = wasmssa.trunc_sat_si %[[CONST_0]] : f64 to i64
// CHECK:           wasmssa.return %[[TRUNC_SAT_SI_0]] : i64
// CHECK:         }

// CHECK-LABEL:   wasmssa.func exported @i64_trunc_sat_f64_u() -> i64 {
// CHECK:           %[[CONST_0:.*]] = wasmssa.const 1.210000e+01 : f64
// CHECK:           %[[TRUNC_SAT_UI_0:.*]] = wasmssa.trunc_sat_ui %[[CONST_0]] : f64 to i64
// CHECK:           wasmssa.return %[[TRUNC_SAT_UI_0]] : i64
// CHECK:         }
