// RUN: yaml2obj %S/inputs/loop_with_inst.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Code used to create this test:

(module
  (func (result i32)
    (local $i i32)
    (loop $my_loop (result i32)
      local.get $i
      i32.const 1
      i32.add
      local.set $i
      local.get $i
      i32.const 10
      i32.lt_s
    )
  )
)*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local of type i32
// CHECK:           wasmssa.loop : {
// CHECK:             %[[VAL_1:.*]] = wasmssa.local_get %[[VAL_0]] :  ref to i32
// CHECK:             %[[VAL_2:.*]] = wasmssa.const 1 : i32
// CHECK:             %[[VAL_3:.*]] = wasmssa.add %[[VAL_1]] %[[VAL_2]] : i32
// CHECK:             wasmssa.local_set %[[VAL_0]] :  ref to i32 to %[[VAL_3]] : i32
// CHECK:             %[[VAL_4:.*]] = wasmssa.local_get %[[VAL_0]] :  ref to i32
// CHECK:             %[[VAL_5:.*]] = wasmssa.const 10 : i32
// CHECK:             %[[VAL_6:.*]] = wasmssa.lt_si %[[VAL_4]] %[[VAL_5]] : i32 -> i32
// CHECK:             wasmssa.block_return %[[VAL_6]] : i32
// CHECK:           }> ^bb1
// CHECK:         ^bb1(%[[VAL_7:.*]]: i32):
// CHECK:           wasmssa.return %[[VAL_7]] : i32
