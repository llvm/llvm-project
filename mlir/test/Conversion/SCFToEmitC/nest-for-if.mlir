// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-emitc %s | FileCheck %s

// CHECK-LABEL: func.func @nest_for_in_if
// CHECK-SAME: %[[ARG_0:.*]]: i1, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index, %[[ARG_3:.*]]: index, %[[ARG_4:.*]]: f32
// CHECK-NEXT:    %[[CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_1]] : index to !emitc.size_t
// CHECK-NEXT:    %[[CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG_2]] : index to !emitc.size_t
// CHECK-NEXT:    %[[CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ARG_3]] : index to !emitc.size_t
// CHECK-NEXT:    emitc.if %[[ARG_0]] {
// CHECK-NEXT:      emitc.for %[[ARG_5:.*]] = %[[CAST_0]] to %[[CAST_1]] step %[[CAST_2]] {
// CHECK-NEXT:        %[[CST_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[CAST_3:.*]] = builtin.unrealized_conversion_cast %[[CST_1]] : index to !emitc.size_t
// CHECK-NEXT:        emitc.for %[[ARG_6:.*]] = %[[CAST_0]] to %[[CAST_1]] step %[[CAST_3]] {
// CHECK-NEXT:          %[[CST_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %3 = emitc.call_opaque "func_false"(%[[ARG_4]]) : (f32) -> i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @nest_for_in_if(%arg0: i1, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) {
  scf.if %arg0 {
    scf.for %i0 = %arg1 to %arg2 step %arg3 {
      %c1 = arith.constant 1 : index
      scf.for %i1 = %arg1 to %arg2 step %c1 {
        %c1_0 = arith.constant 1 : index
      }
    }
  } else {
    %0 = emitc.call_opaque "func_false"(%arg4) : (f32) -> i32
  }
  return
}
