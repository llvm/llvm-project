// RUN: yaml2obj %S/inputs/if.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
(type $intMapper (func (param $input i32) (result i32)))
(func $if_else (type $intMapper)
  local.get 0
  i32.const 1
  i32.and
  if $isOdd (result i32)
    local.get 0
    i32.const 3
    i32.mul
    i32.const 1
    i32.add
  else
    local.get 0
    i32.const 1
    i32.shr_u
  end
)

(func $if_only (type $intMapper)
  local.get 0
  local.get 0
  i32.const 1
  i32.and
  if $isOdd (type $intMapper)
    i32.const 1
    i32.add
  end
)

(func $if_if (type $intMapper)
  local.get 0
  i32.ctz
  if $isEven (result i32)
    i32.const 2
    local.get 0
    i32.const 1
    i32.shr_u
    i32.ctz
    if $isMultipleOfFour (type $intMapper)
      i32.const 2
      i32.add
    end
  else
    i32.const 1
  end
)
)
*/
// CHECK-LABEL:   wasmssa.func @func_0(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.and %[[VAL_0]] %[[VAL_1]] : i32
// CHECK:           wasmssa.if %[[VAL_2]] : {
// CHECK:             %[[VAL_3:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:             %[[VAL_4:.*]] = wasmssa.const 3 : i32
// CHECK:             %[[VAL_5:.*]] = wasmssa.mul %[[VAL_3]] %[[VAL_4]] : i32
// CHECK:             %[[VAL_6:.*]] = wasmssa.const 1 : i32
// CHECK:             %[[VAL_7:.*]] = wasmssa.add %[[VAL_5]] %[[VAL_6]] : i32
// CHECK:             wasmssa.block_return %[[VAL_7]] : i32
// CHECK:           } "else "{
// CHECK:             %[[VAL_8:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:             %[[VAL_9:.*]] = wasmssa.const 1 : i32
// CHECK:             %[[VAL_10:.*]] = wasmssa.shr_u %[[VAL_8]] by %[[VAL_9]] bits : i32
// CHECK:             wasmssa.block_return %[[VAL_10]] : i32
// CHECK:           }> ^bb1
// CHECK:         ^bb1(%[[VAL_11:.*]]: i32):
// CHECK:           wasmssa.return %[[VAL_11]] : i32

// CHECK-LABEL:   wasmssa.func @func_1(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.const 1 : i32
// CHECK:           %[[VAL_3:.*]] = wasmssa.and %[[VAL_1]] %[[VAL_2]] : i32
// CHECK:           wasmssa.if %[[VAL_3]](%[[VAL_0]]) : i32 : {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32):
// CHECK:             %[[VAL_5:.*]] = wasmssa.const 1 : i32
// CHECK:             %[[VAL_6:.*]] = wasmssa.add %[[VAL_4]] %[[VAL_5]] : i32
// CHECK:             wasmssa.block_return %[[VAL_6]] : i32
// CHECK:           } > ^bb1
// CHECK:         ^bb1(%[[VAL_7:.*]]: i32):
// CHECK:           wasmssa.return %[[VAL_7]] : i32

// CHECK-LABEL:   wasmssa.func @func_2(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.ctz %[[VAL_0]] : i32
// CHECK:           wasmssa.if %[[VAL_1]] : {
// CHECK:             %[[VAL_2:.*]] = wasmssa.const 2 : i32
// CHECK:             %[[VAL_3:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:             %[[VAL_4:.*]] = wasmssa.const 1 : i32
// CHECK:             %[[VAL_5:.*]] = wasmssa.shr_u %[[VAL_3]] by %[[VAL_4]] bits : i32
// CHECK:             %[[VAL_6:.*]] = wasmssa.ctz %[[VAL_5]] : i32
// CHECK:             wasmssa.if %[[VAL_6]](%[[VAL_2]]) : i32 : {
// CHECK:             ^bb0(%[[VAL_7:.*]]: i32):
// CHECK:               %[[VAL_8:.*]] = wasmssa.const 2 : i32
// CHECK:               %[[VAL_9:.*]] = wasmssa.add %[[VAL_7]] %[[VAL_8]] : i32
// CHECK:               wasmssa.block_return %[[VAL_9]] : i32
// CHECK:             } > ^bb1
// CHECK:           ^bb1(%[[VAL_10:.*]]: i32):
// CHECK:             wasmssa.block_return %[[VAL_10]] : i32
// CHECK:           } "else "{
// CHECK:             %[[VAL_11:.*]] = wasmssa.const 1 : i32
// CHECK:             wasmssa.block_return %[[VAL_11]] : i32
// CHECK:           }> ^bb1
// CHECK:         ^bb1(%[[VAL_12:.*]]: i32):
// CHECK:           wasmssa.return %[[VAL_12]] : i32
