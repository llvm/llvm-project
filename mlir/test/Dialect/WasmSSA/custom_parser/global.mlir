// RUN: mlir-opt %s | FileCheck %s

module {
  wasmssa.import_global "from_js" from "env" as @global_0 : i32

  wasmssa.global @global_1 i32 : {
    %0 = wasmssa.const 10 : i32
    wasmssa.return %0 : i32
  }
  wasmssa.global @global_2 i32 mutable : {
    %0 = wasmssa.const 17 : i32
    wasmssa.return %0 : i32
  }
  wasmssa.global @global_3 i32 mutable : {
    %0 = wasmssa.const 10 : i32
    wasmssa.return %0 : i32
  }
  wasmssa.global @global_4 i32 : {
    %0 = wasmssa.global_get @global_0 : i32
    wasmssa.return %0 : i32
  }
}

// CHECK-LABEL:   wasmssa.import_global "from_js" from "env" as @global_0 : i32

// CHECK-LABEL:   wasmssa.global @global_1 i32 : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.global @global_2 i32 mutable : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 17 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.global @global_3 i32 mutable : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.global @global_4 i32 : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.global_get @global_0 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32
// CHECK:         }
