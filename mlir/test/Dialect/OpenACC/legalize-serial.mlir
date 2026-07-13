// RUN: mlir-opt %s -acc-legalize-serial | FileCheck %s

acc.private.recipe @privatization_memref_10_f32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

acc.private.recipe @privatization_memref_10_10_f32 : memref<10x10xf32> init {
^bb0(%arg0: memref<10x10xf32>):
  %0 = memref.alloc() : memref<10x10xf32>
  acc.yield %0 : memref<10x10xf32>
} destroy {
^bb0(%arg0: memref<10x10xf32>):
  memref.dealloc %arg0 : memref<10x10xf32> 
  acc.terminator
}

acc.firstprivate.recipe @firstprivatization_memref_10xf32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} copy {
^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
  acc.terminator
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

acc.reduction.recipe @reduction_add_i64 : i64 reduction_operator<add> init {
^bb0(%0: i64):
  %1 = arith.constant 0 : i64
  acc.yield %1 : i64
} combiner {
^bb0(%0: i64, %1: i64):
  %2 = arith.addi %0, %1 : i64
  acc.yield %2 : i64
}

acc.reduction.recipe @reduction_add_memref_i64 : memref<i64> reduction_operator<add> init {
^bb0(%arg0: memref<i64>):
  %0 = memref.alloca() : memref<i64>
  %c0 = arith.constant 0 : i64
  memref.store %c0, %0[] : memref<i64>
  acc.yield %0 : memref<i64>
} combiner {
^bb0(%arg0: memref<i64>, %arg1: memref<i64>):
  %0 = memref.load %arg0[] : memref<i64>
  %1 = memref.load %arg1[] : memref<i64>
  %2 = arith.addi %0, %1 : i64
  memref.store %2, %arg0[] : memref<i64>
  acc.terminator
}

// CHECK:   func.func @testserialop(%[[VAL_0:.*]]: memref<10xf32>, %[[VAL_1:.*]]: memref<10xf32>, %[[VAL_2:.*]]: memref<10x10xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           acc.parallel async(%[[VAL_3]] : i64) num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           }
// CHECK:           acc.parallel async(%[[VAL_4]] : i32) num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           }
// CHECK:           acc.parallel async(%[[VAL_5]] : index) num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           }
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) wait({%[[VAL_3]] : i64}) {
// CHECK:           }
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) wait({%[[VAL_4]] : i32}) {
// CHECK:           }
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) wait({%[[VAL_5]] : index}) {
// CHECK:           }
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) wait({%[[VAL_3]] : i64, %[[VAL_4]] : i32, %[[VAL_5]] : index}) {
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = acc.firstprivate varPtr(%[[VAL_1]] : memref<10xf32>) recipe(@firstprivatization_memref_10xf32) -> memref<10xf32>
// CHECK:           %[[VAL_9:.*]] = acc.private varPtr(%[[VAL_2]] : memref<10x10xf32>) recipe(@privatization_memref_10_10_f32) -> memref<10x10xf32>
// CHECK:           acc.parallel firstprivate(%[[VAL_6]] : memref<10xf32>) num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) private(%[[VAL_9]] : memref<10x10xf32>) vector_length(%[[VAL_4]] : i32) {
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = acc.copyin varPtr(%[[VAL_0]] : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
// CHECK:           acc.parallel dataOperands(%[[VAL_7]] : memref<10xf32>) num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           }
// CHECK:           %[[I64MEM:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[VAL_3]], %[[I64MEM]][] : memref<i64>
// CHECK:           %[[VAL_10:.*]] = acc.reduction varPtr(%[[I64MEM]] : memref<i64>) recipe(@reduction_add_memref_i64) -> memref<i64>
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) reduction(%[[VAL_10]] : memref<i64>) {
// CHECK:           }
// CHECK:           acc.parallel combined(loop) num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:             acc.loop combined(serial) control(%{{.*}} : index) = (%[[VAL_5]] : index) to (%[[VAL_5]] : index) step (%[[VAL_5]] : index) {
// CHECK:               acc.yield
// CHECK:             } attributes {seq = [#acc.device_type<none>]}
// CHECK:             acc.terminator
// CHECK:           }
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           } attributes {defaultAttr = #acc<defaultvalue none>}
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           } attributes {defaultAttr = #acc<defaultvalue present>}
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           }
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           }
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:           } attributes {selfAttr}
// CHECK:           acc.parallel num_gangs({%[[VAL_4]] : i32}) num_workers(%[[VAL_4]] : i32) vector_length(%[[VAL_4]] : i32) {
// CHECK:             acc.yield
// CHECK:           } attributes {selfAttr}
// CHECK:           return
// CHECK:         }

func.func @testserialop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %i64value = arith.constant 1 : i64
  %i32value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index
  acc.serial async(%i64value: i64) {
  }
  acc.serial async(%i32value: i32) {
  }
  acc.serial async(%idxValue: index) {
  }
  acc.serial wait({%i64value: i64}) {
  }
  acc.serial wait({%i32value: i32}) {
  }
  acc.serial wait({%idxValue: index}) {
  }
  acc.serial wait({%i64value : i64, %i32value : i32, %idxValue : index}) {
  }
  %firstprivate = acc.firstprivate varPtr(%b : memref<10xf32>) recipe(@firstprivatization_memref_10xf32) -> memref<10xf32>
  %c_private = acc.private varPtr(%c : memref<10x10xf32>) recipe(@privatization_memref_10_10_f32) -> memref<10x10xf32>
  acc.serial private(%c_private : memref<10x10xf32>) firstprivate(%firstprivate : memref<10xf32>) {
  }
  %copyinfromcopy = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
  acc.serial dataOperands(%copyinfromcopy : memref<10xf32>) {
  }
  %i64mem = memref.alloca() : memref<i64>
  memref.store %i64value, %i64mem[] : memref<i64>
  %i64reduction = acc.reduction varPtr(%i64mem : memref<i64>) recipe(@reduction_add_memref_i64) -> memref<i64>
  acc.serial reduction(%i64reduction : memref<i64>) {
  }
  acc.serial combined(loop) {
    acc.loop combined(serial) control(%arg3 : index) = (%idxValue : index) to (%idxValue : index) step (%idxValue : index) {
      acc.yield
    } attributes {seq = [#acc.device_type<none>]}
    acc.terminator
  }
  acc.serial {
  } attributes {defaultAttr = #acc<defaultvalue none>}
  acc.serial {
  } attributes {defaultAttr = #acc<defaultvalue present>}
  acc.serial {
  } attributes {asyncAttr}
  acc.serial {
  } attributes {waitAttr}
  acc.serial {
  } attributes {selfAttr}
  acc.serial {
    acc.yield
  } attributes {selfAttr}
  return
}

