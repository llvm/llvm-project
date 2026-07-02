// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s
// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir --canonicalize -o - | FileCheck --check-prefix=CHECK-CANONICAL %s

module {
  wasmssa.func @func_0() {
    wasmssa.loop : {
      wasmssa.block_return
    }> ^bb1
  ^bb1:  // pred: ^bb0
    wasmssa.return
  }
}

// CHECK-LABEL:   module {
// CHECK:           func.func @func_0() {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             return
// CHECK:           }

// CHECK-CANONICAL-LABEL:  func.func @func_0() {
// CHECK-CANONICAL:             return
// CHECK-CANONICAL:           }

// -----

module {
  wasmssa.func @func_0() -> i32 {
    %0 = wasmssa.local of type i32
    wasmssa.loop : {
      %1 = wasmssa.local_get %0 : ref to i32
      %2 = wasmssa.const 10 : i32
      %3 = wasmssa.lt_si %1 %2 : i32 -> i32
      wasmssa.block_return %3 : i32
    }> ^bb1
  ^bb1(%1: i32):  // pred: ^bb0
    wasmssa.return %1 : i32
  }
}
// CHECK-LABEL:   func.func @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           memref.store %[[VAL_1]], %[[VAL_0]][] : memref<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_4:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.extui %[[VAL_4]] : i1 to i32
// CHECK:           cf.br ^bb2(%[[VAL_5]] : i32)
// CHECK:         ^bb2(%[[VAL_6:.*]]: i32):
// CHECK:           return %[[VAL_6]] : i32
// CHECK:         }

// CHECK-CANONICAL-LABEL:   func.func @func_0() -> i32 {
// CHECK-CANONICAL:           %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK-CANONICAL:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-CANONICAL:           %[[VAL_2:.*]] = memref.alloca() : memref<i32>
// CHECK-CANONICAL:           memref.store %[[VAL_1]], %[[VAL_2]][] : memref<i32>
// CHECK-CANONICAL:           %[[VAL_3:.*]] = memref.load %[[VAL_2]][] : memref<i32>
// CHECK-CANONICAL:           %[[VAL_4:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VAL_0]] : i32
// CHECK-CANONICAL:           %[[VAL_5:.*]] = arith.extui %[[VAL_4]] : i1 to i32
// CHECK-CANONICAL:           return %[[VAL_5]] : i32

// -----

module {
  wasmssa.func @func_0() {
    %0 = wasmssa.local of type i32
    wasmssa.loop : {
      %1 = wasmssa.local_get %0 : ref to i32
      %2 = wasmssa.const 10 : i32
      %3 = wasmssa.lt_si %1 %2 : i32 -> i32
      wasmssa.branch_if %3 to level 0 else ^bb1
    ^bb1:  // pred: ^bb0
      wasmssa.block_return
    }> ^bb1
  ^bb1:  // pred: ^bb0
    wasmssa.return
  }
}

// CHECK-LABEL:   func.func @func_0() {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           memref.store %[[VAL_1]], %[[VAL_0]][] : memref<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_4:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.extui %[[VAL_4]] : i1 to i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           cf.cond_br %[[VAL_7]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return
// CHECK:         }

// CHECK-CANONICAL-LABEL:   func.func @func_0() {
// CHECK-CANONICAL:           %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK-CANONICAL:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-CANONICAL:           %[[VAL_2:.*]] = memref.alloca() : memref<i32>
// CHECK-CANONICAL:           memref.store %[[VAL_1]], %[[VAL_2]][] : memref<i32>
// CHECK-CANONICAL:           cf.br ^bb1
// CHECK-CANONICAL:         ^bb1:
// CHECK-CANONICAL:           %[[VAL_3:.*]] = memref.load %[[VAL_2]][] : memref<i32>
// CHECK-CANONICAL:           %[[VAL_4:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VAL_0]] : i32
// CHECK-CANONICAL:           cf.cond_br %[[VAL_4]], ^bb1, ^bb2
// CHECK-CANONICAL:         ^bb2:
// CHECK-CANONICAL:           return

// -----

module {
  wasmssa.func @func_0() {
    %0 = wasmssa.local of type i32
    %1 = wasmssa.local of type i32
    wasmssa.loop : {
      %2 = wasmssa.local_get %0 : ref to i32
      %3 = wasmssa.const 1 : i32
      %4 = wasmssa.add %2 %3 : i32
      wasmssa.local_set %0 : ref to i32 to %4 : i32
      wasmssa.loop : {
        %8 = wasmssa.const 12 : i32
        %9 = wasmssa.local_get %0 : ref to i32
        %10 = wasmssa.gt_si %8 %9 : i32 -> i32
        wasmssa.branch_if %10 to level 0 else ^bb1
      ^bb1:  // pred: ^bb0
        wasmssa.block_return %8 : i32
      }> ^bb1
    ^bb1(%5: i32):  // pred: ^bb0
      %6 = wasmssa.const 10 : i32
      %7 = wasmssa.lt_si %5 %6 : i32 -> i32
      wasmssa.branch_if %7 to level 0 else ^bb2
    ^bb2:  // pred: ^bb1
      wasmssa.block_return
    }> ^bb1
  ^bb1:  // pred: ^bb0
    wasmssa.return
  }
}

// CHECK-LABEL:   func.func @func_0() {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           memref.store %[[VAL_1]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = memref.alloca() : memref<i32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           memref.store %[[VAL_3]], %[[VAL_2]][] : memref<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_4:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:           memref.store %[[VAL_6]], %[[VAL_0]][] : memref<i32>
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_7:.*]] = arith.constant 12 : i32
// CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.extui %[[VAL_9]] : i1 to i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:           cf.cond_br %[[VAL_12]], ^bb2, ^bb3
// CHECK:         ^bb3:
// CHECK:           cf.br ^bb4(%[[VAL_7]] : i32)
// CHECK:         ^bb4(%[[VAL_13:.*]]: i32):
// CHECK:           %[[VAL_14:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_16:.*]] = arith.extui %[[VAL_15]] : i1 to i32
// CHECK:           %[[VAL_17:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_17]] : i32
// CHECK:           cf.cond_br %[[VAL_18]], ^bb1, ^bb5
// CHECK:         ^bb5:
// CHECK:           cf.br ^bb6
// CHECK:         ^bb6:
// CHECK:           return
// CHECK:         }

// CHECK-CANONICAL-LABEL:   func.func @func_0() {
// CHECK-CANONICAL:           %[[CONSTANT_0:.*]] = arith.constant 12 : i32
// CHECK-CANONICAL:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-CANONICAL:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK-CANONICAL:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<i32>
// CHECK-CANONICAL:           memref.store %[[CONSTANT_2]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-CANONICAL:           %[[LOAD_0:.*]] = memref.load %[[ALLOCA_0]][] : memref<i32>
// CHECK-CANONICAL:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_0]], %[[CONSTANT_1]] : i32
// CHECK-CANONICAL:           memref.store %[[ADDI_0]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-CANONICAL:           cf.br ^bb1
// CHECK-CANONICAL:         ^bb1:
// CHECK-CANONICAL:           %[[LOAD_1:.*]] = memref.load %[[ALLOCA_0]][] : memref<i32>
// CHECK-CANONICAL:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_1]], %[[CONSTANT_0]] : i32
// CHECK-CANONICAL:           cf.cond_br %[[CMPI_0]], ^bb1, ^bb2
// CHECK-CANONICAL:         ^bb2:
// CHECK-CANONICAL:           return
// CHECK-CANONICAL:         }
