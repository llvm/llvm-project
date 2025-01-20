// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-emitc %s | FileCheck %s

func.func @simple_std_for_loop(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = arith.constant 1 : index
  }
  return
}
// CHECK-LABEL: func.func @simple_std_for_loop(
// CHECK-SAME:      %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index) {
// CHECK-NEXT:    %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[ARG_2]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[ARG_1]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK-NEXT:    emitc.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] : !emitc.size_t {
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @simple_std_2_for_loops(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = arith.constant 1 : index
    scf.for %i1 = %arg0 to %arg1 step %arg2 {
      %c1_0 = arith.constant 1 : index
    }
  }
  return
}
// CHECK-LABEL: func.func @simple_std_2_for_loops(
// CHECK-SAME:      %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index) {
// CHECK-NEXT:    %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[ARG_2]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[ARG_1]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK-NEXT:    emitc.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] : !emitc.size_t {
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:      emitc.for %[[VAL_5:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] : !emitc.size_t {
// CHECK-NEXT:        %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @for_yield(%arg0 : index, %arg1 : index, %arg2 : index) -> (f32, f32) {
  %s0 = arith.constant 0.0 : f32
  %s1 = arith.constant 1.0 : f32
  %result:2 = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0, %sj = %s1) -> (f32, f32) {
    %sn = arith.addf %si, %sj : f32
    scf.yield %sn, %sn : f32, f32
  }
  return %result#0, %result#1 : f32, f32
}
// CHECK-LABEL: func.func @for_yield(
// CHECK-SAME:      %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index) -> (f32, f32) {
// CHECK-NEXT:    %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[ARG_2]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[ARG_1]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[VAL_5:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
// CHECK-NEXT:    %[[VAL_6:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
// CHECK-NEXT:    emitc.assign %[[VAL_3]] : f32 to %[[VAL_5]] : <f32>
// CHECK-NEXT:    emitc.assign %[[VAL_4]] : f32 to %[[VAL_6]] : <f32>
// CHECK-NEXT:    emitc.for %[[VAL_7:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] : !emitc.size_t {
// CHECK-NEXT:      %[[VAL_8:.*]] = emitc.load %[[VAL_5]] : <f32>
// CHECK-NEXT:      %[[VAL_9:.*]] = emitc.load %[[VAL_6]] : <f32>
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f32
// CHECK-NEXT:      emitc.assign %[[VAL_10]] : f32 to %[[VAL_5]] : <f32>
// CHECK-NEXT:      emitc.assign %[[VAL_10]] : f32 to %[[VAL_6]] : <f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[VAL_11:.*]] = emitc.load %[[VAL_5]] : <f32>
// CHECK-NEXT:    %[[VAL_12:.*]] = emitc.load %[[VAL_6]] : <f32>
// CHECK-NEXT:    return %[[VAL_11]], %[[VAL_12]] : f32, f32
// CHECK-NEXT:  }

func.func @nested_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) -> f32 {
  %s0 = arith.constant 1.0 : f32
  %r = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%iter = %s0) -> (f32) {
    %result = scf.for %i1 = %arg0 to %arg1 step %arg2 iter_args(%si = %iter) -> (f32) {
      %sn = arith.addf %si, %si : f32
      scf.yield %sn : f32
    }
    scf.yield %result : f32
  }
  return %r : f32
}
// CHECK-LABEL: func.func @nested_for_yield(
// CHECK-SAME:      %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index) -> f32 {
// CHECK-NEXT:    %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[ARG_2]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[ARG_1]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK-NEXT:    %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
// CHECK-NEXT:    emitc.assign %[[VAL_3]] : f32 to %[[VAL_4]] : <f32>
// CHECK-NEXT:    emitc.for %[[VAL_5:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] : !emitc.size_t {
// CHECK-NEXT:      %[[VAL_6:.*]] = emitc.load %[[VAL_4]] : <f32>
// CHECK-NEXT:      %[[VAL_7:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
// CHECK-NEXT:      emitc.assign %[[VAL_6]] : f32 to %[[VAL_7]] : <f32>
// CHECK-NEXT:      emitc.for %[[VAL_8:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] : !emitc.size_t {
// CHECK-NEXT:        %[[VAL_9:.*]] = emitc.load %[[VAL_7]] : <f32>  
// CHECK-NEXT:        %[[VAL_10:.*]] = arith.addf %[[VAL_9]], %[[VAL_9]] : f32
// CHECK-NEXT:        emitc.assign %[[VAL_10]] : f32 to %[[VAL_7]] : <f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[VAL_11:.*]] = emitc.load %[[VAL_7]] : <f32>  
// CHECK-NEXT:      emitc.assign %[[VAL_11]] : f32 to %[[VAL_4]] : <f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[VAL_12:.*]] = emitc.load %[[VAL_4]] : <f32>  
// CHECK-NEXT:    return %[[VAL_12]] : f32
// CHECK-NEXT:  }

func.func @for_yield_index(%arg0 : index, %arg1 : index, %arg2 : index) -> index {
  %zero = arith.constant 0 : index
  %r = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%acc = %zero) -> index {
    scf.yield %acc : index
  }
  return %r : index
}

// CHECK-LABEL: func.func @for_yield_index(
// CHECK-SAME: %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index) -> index {
// CHECK:     %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_2]] : index to !emitc.size_t
// CHECK:     %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[ARG_1]] : index to !emitc.size_t
// CHECK:     %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:     %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to !emitc.size_t
// CHECK:     %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.size_t>
// CHECK:     emitc.assign %[[VAL_3]] : !emitc.size_t to %[[VAL_4]] : <!emitc.size_t>
// CHECK:     emitc.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_1]] step %[[VAL_0]] : !emitc.size_t {
// CHECK:       %[[V:.*]] = emitc.load %[[VAL_4]] : <!emitc.size_t>
// CHECK:       emitc.assign %[[V]] : !emitc.size_t to %[[VAL_4]] : <!emitc.size_t>
// CHECK:     }
// CHECK:     %[[V2:.*]] = emitc.load %[[VAL_4]] : <!emitc.size_t>
// CHECK:     %[[VAL_8:.*]] = builtin.unrealized_conversion_cast %[[V2]] : !emitc.size_t to index
// CHECK:     return %[[VAL_8]] : index
// CHECK:   }


func.func @for_yield_update_loop_carried_var(%arg0 : index, %arg1 : index, %arg2 : index) -> index {
   %zero = arith.constant 0 : index
   %r = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%acc = %zero) -> index {
     %sn = arith.addi %acc, %acc : index
     scf.yield %sn: index
   }
   return %r : index
 }

// CHECK-LABEL: func.func @for_yield_update_loop_carried_var(
// CHECK-SAME:  %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index) -> index {
// CHECK:   %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_2]] : index to !emitc.size_t
// CHECK:   %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[ARG_1]] : index to !emitc.size_t
// CHECK:   %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to !emitc.size_t
// CHECK:   %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.size_t>
// CHECK:   emitc.assign %[[VAL_3]] : !emitc.size_t to %[[VAL_4]] : <!emitc.size_t>
// CHECK:   emitc.for %[[ARG_3:.*]] = %[[VAL_2]] to %[[VAL_1]] step %[[VAL_0]] : !emitc.size_t {
// CHECK:     %[[V:.*]] = emitc.load %[[VAL_4]] : <!emitc.size_t>
// CHECK:     %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[V]] : !emitc.size_t to index
// CHECK:     %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_5]] : index
// CHECK:     %[[VAL_8:.*]] = builtin.unrealized_conversion_cast %[[VAL_6]] : index to !emitc.size_t
// CHECK:     emitc.assign %[[VAL_8]] : !emitc.size_t to %[[VAL_4]] : <!emitc.size_t>
// CHECK:   }
// CHECK:   %[[V2:.*]] = emitc.load %[[VAL_4]] : <!emitc.size_t>
// CHECK:   %[[VAL_9:.*]] = builtin.unrealized_conversion_cast %[[V2]] : !emitc.size_t to index
// CHECK:   return %[[VAL_9]] : index
// CHECK: }
