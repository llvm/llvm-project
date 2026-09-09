// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s
// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir --canonicalize -o - | FileCheck --check-prefix=CHECK_CANONICALIZED %s

// CHECK-LABEL:   func.func @i_am_a_block(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = arith.constant 17 : i32
// CHECK:           memref.store %[[VAL_1]], %[[VAL_0]][] : memref<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : i32
// CHECK:           memref.store %[[VAL_2]], %[[VAL_0]][] : memref<i32>
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           return %[[VAL_3]] : i32

// CHECK_CANONICALIZED-LABEL:   func.func @i_am_a_block(
// CHECK_CANONICALIZED-SAME:      %[[VAL_0:.*]]: i32) -> i32 {
// CHECK_CANONICALIZED:           %[[VAL_1:.*]] = arith.constant 42 : i32
// CHECK_CANONICALIZED:           %[[VAL_3:.*]] = memref.alloca() : memref<i32>
// CHECK_CANONICALIZED:           memref.store %[[VAL_1]], %[[VAL_3]][] : memref<i32>
// CHECK_CANONICALIZED-NOT:       memref.store
// CHECK_CANONICALIZED:           %[[VAL_4:.*]] = memref.load %[[VAL_3]][] : memref<i32>
// CHECK_CANONICALIZED:           return %[[VAL_4]] : i32
wasmssa.func @i_am_a_block(%arg0 : !wasmssa<local ref to i32>) -> i32 {
  %1 = wasmssa.const 17 : i32
  wasmssa.local_set %arg0 : ref to i32 to %1 : i32
  wasmssa.block : {
    %2 = wasmssa.const 42 : i32
    wasmssa.local_set %arg0 : ref to i32 to %2 : i32
    wasmssa.block_return
  }> ^bb1
  ^bb1:
  %res = wasmssa.local_get %arg0 : ref to i32
  wasmssa.return %res : i32
}


// CHECK-LABEL:   func.func @func_0(
// CHECK-SAME:                      %[[ARG0:.*]]: i32) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : i32
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : i32
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           %[[VAL_4:.*]] = arith.constant 5 : i32
// CHECK:           cf.br ^bb5
// CHECK:         ^bb5:
// CHECK:           %[[VAL_5:.*]] = arith.constant 6 : i32
// CHECK:           cf.br ^bb6
// CHECK:         ^bb6:
// CHECK:           %[[VAL_6:.*]] = arith.constant 7 : i32
// CHECK:           return
wasmssa.func @func_0(%arg0: !wasmssa<local ref to i32>) {
  %1 = wasmssa.const 1: i32
  wasmssa.block : {
    %2 = wasmssa.const 2: i32
    wasmssa.block : {
      %3 = wasmssa.const 3: i32
      wasmssa.block : {
        %4 = wasmssa.const 4: i32
        wasmssa.block_return
      }> ^bb1
    ^bb1:  // pred: ^bb0
      %5 = wasmssa.const 5: i32
      wasmssa.block_return
    }> ^bb1
  ^bb1:  // pred: ^bb0
    %6 = wasmssa.const 6: i32
    wasmssa.block_return
  }> ^bb1
^bb1:  // pred: ^bb0
  %7 = wasmssa.const 7: i32
  wasmssa.return
}

// CHECK-LABEL:   func.func @func_1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : i32
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : i32
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           %[[VAL_4:.*]] = arith.constant 5 : i32
// CHECK:           cf.br ^bb5
// CHECK:         ^bb5:
// CHECK:           %[[VAL_5:.*]] = arith.constant 6 : i32
// CHECK:           cf.br ^bb6
// CHECK:         ^bb6:
// CHECK:           %[[VAL_6:.*]] = arith.constant 7 : i32
// CHECK:           return
wasmssa.func @func_1() {
  %1 = wasmssa.const 1: i32
  wasmssa.block : {
    %2 = wasmssa.const 2: i32
    wasmssa.block_return
  }> ^bb1
^bb1:  // pred: ^bb0
  %3 = wasmssa.const 3: i32
  wasmssa.block : {
    %4 = wasmssa.const 4: i32
    wasmssa.block_return
  }> ^bb2
^bb2:  // pred: ^bb1
  %5 = wasmssa.const 5: i32
  wasmssa.block : {
    %6 = wasmssa.const 6: i32
    wasmssa.block_return
  }> ^bb3
^bb3:  // pred: ^bb2
  %7 = wasmssa.const 7: i32
  wasmssa.return
}

// CHECK-LABEL:   func.func @func_2() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 14 : i32
// CHECK:           cf.br ^bb1(%[[VAL_0]] : i32)
// CHECK:         ^bb1(%[[VAL_1:.*]]: i32):
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           cf.br ^bb2(%[[VAL_3]] : i32)
// CHECK:         ^bb2(%[[VAL_4:.*]]: i32):
// CHECK:           return %[[VAL_4]] : i32
wasmssa.func @func_2() -> i32 {
  %0 = wasmssa.const 14 : i32
  wasmssa.block(%0) : i32 : {
  ^bb0(%arg0: i32):
    %2 = wasmssa.const 1 : i32
    %3 = wasmssa.add %arg0 %2 : i32
    wasmssa.block_return %3 : i32
  }> ^bb1
^bb1(%arg0: i32):
  wasmssa.return %arg0 : i32
}

// CHECK-LABEL:   func.func @func_3() -> i32 {
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_0:.*]] = arith.constant 17 : i32
// CHECK:           cf.br ^bb2(%[[VAL_0]] : i32)
// CHECK:         ^bb2(%[[VAL_1:.*]]: i32):
// CHECK:           return %[[VAL_1]] : i32
wasmssa.func @func_3() -> i32 {
  wasmssa.block : {
    %1 = wasmssa.const 17 : i32
    wasmssa.block_return %1 : i32
  }> ^bb1
^bb1(%arg0: i32):
  wasmssa.return %arg0 : i32
}

//// ============= Branch instructions etc ==========

// CHECK-LABEL:   func.func @branch_if_taken() -> i32 {
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.cmpi ne, %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           cf.cond_br %[[VAL_3]], ^bb3(%[[VAL_0]] : i32), ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_4:.*]] = arith.constant 16 : i32
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %[[VAL_4]] : i32
// CHECK:           cf.br ^bb3(%[[VAL_5]] : i32)
// CHECK:         ^bb3(%[[VAL_6:.*]]: i32):
// CHECK:           return %[[VAL_6]] : i32

// CHECK_CANONICALIZED-LABEL:   func.func @branch_if_taken() -> i32 {
// CHECK_CANONICALIZED:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK_CANONICALIZED:           return %[[VAL_0]] : i32

wasmssa.func @branch_if_taken() -> i32 {
  wasmssa.block : {
    %1 = wasmssa.const 1 : i32
    %2 = wasmssa.const 2 : i32
    wasmssa.branch_if %2 to level 0 with args(%1 : i32) else ^bb1
  ^bb1:  // pred: ^bb0
    %3 = wasmssa.const 16 : i32
    %4 = wasmssa.add %1 %3 : i32
    wasmssa.block_return %4 : i32
  }> ^bb1
^bb1(%0: i32):  // pred: ^bb0
  wasmssa.return %0 : i32
}

// CHECK-LABEL:   func.func @branch_if_continue() -> i32 {
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.cmpi ne, %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           cf.cond_br %[[VAL_3]], ^bb3(%[[VAL_0]] : i32), ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_4:.*]] = arith.constant 16 : i32
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %[[VAL_4]] : i32
// CHECK:           cf.br ^bb3(%[[VAL_5]] : i32)
// CHECK:         ^bb3(%[[VAL_6:.*]]: i32):
// CHECK:           return %[[VAL_6]] : i32

// CHECK_CANONICALIZED-LABEL:   func.func @branch_if_continue() -> i32 {
// CHECK_CANONICALIZED:           %[[VAL_0:.*]] = arith.constant 17 : i32
// CHECK_CANONICALIZED:           return %[[VAL_0]] : i32
// CHECK_CANONICALIZED:         }
wasmssa.func @branch_if_continue() -> i32 {
  wasmssa.block : {
    %1 = wasmssa.const 1 : i32
    %2 = wasmssa.const 0 : i32
    wasmssa.branch_if %2 to level 0 with args(%1 : i32) else ^bb1
  ^bb1:  // pred: ^bb0
    %3 = wasmssa.const 16 : i32
    %4 = wasmssa.add %1 %3 : i32
    wasmssa.block_return %4 : i32
  }> ^bb1
^bb1(%0: i32):  // pred: ^bb0
  wasmssa.return %0 : i32
}

// CHECK-LABEL:   func.func @if(
// CHECK-SAME:                  %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = arith.andi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i32
// CHECK:           cf.cond_br %[[VAL_5]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_6:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           cf.br ^bb3(%[[VAL_10]] : i32)
// CHECK:         ^bb2:
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_13:.*]] = arith.shrui %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:           cf.br ^bb3(%[[VAL_13]] : i32)
// CHECK:         ^bb3(%[[VAL_14:.*]]: i32):
// CHECK:           return %[[VAL_14]] : i32
wasmssa.func @if(%arg0: !wasmssa<local ref to i32>) -> i32 {
  %1 = wasmssa.local_get %arg0 : ref to i32
  %2 = wasmssa.const 1 : i32
  %3 = wasmssa.and %1 %2 : i32
  "wasmssa.if"(%3)[^bb1] ({
    %5 = wasmssa.local_get %arg0 : ref to i32
    %6 = wasmssa.const 3 : i32
    %7 = wasmssa.mul %5 %6 : i32
    %8 = wasmssa.const 1 : i32
    %9 = wasmssa.add %7 %8 : i32
    wasmssa.block_return %9 : i32
  }, {
    %5 = wasmssa.local_get %arg0 : ref to i32
    %6 = wasmssa.const 1 : i32
    %7 = wasmssa.shr_u %5 by %6 bits : i32
    wasmssa.block_return %7 : i32
  }) : (i32) -> ()
^bb1(%4: i32):  // pred: ^bb0
  wasmssa.return %4 : i32
}

// CHECK-LABEL:   func.func @if_else(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.andi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:           cf.cond_br %[[VAL_6]], ^bb1(%[[VAL_1]] : i32), ^bb2(%[[VAL_1]] : i32)
// CHECK:         ^bb1(%[[VAL_7:.*]]: i32):
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:           cf.br ^bb2(%[[VAL_9]] : i32)
// CHECK:         ^bb2(%[[VAL_10:.*]]: i32):
// CHECK:           return %[[VAL_10]] : i32
wasmssa.func @if_else(%arg0: !wasmssa<local ref to i32>) -> i32 {
  %1 = wasmssa.local_get %arg0 : ref to i32
  %2 = wasmssa.local_get %arg0 : ref to i32
  %3 = wasmssa.const 1 : i32
  %4 = wasmssa.and %2 %3 : i32
  "wasmssa.if"(%4, %1)[^bb1] ({
  ^bb0(%arg1: i32):
    %6 = wasmssa.const 1 : i32
    %7 = wasmssa.add %arg1 %6 : i32
    wasmssa.block_return %7 : i32
  }, {
  }) : (i32, i32) -> ()
^bb1(%5: i32):  // pred: ^bb0
  wasmssa.return %5 : i32
}

// CHECK-LABEL:   func.func @if_if(
// CHECK-SAME:                     %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = math.cttz %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.cmpi ne, %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           cf.cond_br %[[VAL_4]], ^bb1, ^bb4
// CHECK:         ^bb1:
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_6:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.shrui %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]] = math.cttz %[[VAL_8]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:           cf.cond_br %[[VAL_11]], ^bb2(%[[VAL_5]] : i32), ^bb3(%[[VAL_5]] : i32)
// CHECK:         ^bb2(%[[VAL_12:.*]]: i32):
// CHECK:           %[[VAL_13:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:           cf.br ^bb3(%[[VAL_14]] : i32)
// CHECK:         ^bb3(%[[VAL_15:.*]]: i32):
// CHECK:           cf.br ^bb5(%[[VAL_15]] : i32)
// CHECK:         ^bb4:
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:           cf.br ^bb5(%[[VAL_16]] : i32)
// CHECK:         ^bb5(%[[VAL_17:.*]]: i32):
// CHECK:           return %[[VAL_17]] : i32
wasmssa.func @if_if(%arg0: !wasmssa<local ref to i32>) -> i32 {
  %1 = wasmssa.local_get %arg0 : ref to i32
  %2 = wasmssa.ctz %1 : i32
  "wasmssa.if"(%2)[^bb1] ({
    %4 = wasmssa.const 2 : i32
    %5 = wasmssa.local_get %arg0 : ref to i32
    %6 = wasmssa.const 1 : i32
    %7 = wasmssa.shr_u %5 by %6 bits : i32
    %8 = wasmssa.ctz %7 : i32
    "wasmssa.if"(%8, %4)[^bb1] ({
    ^bb0(%arg1: i32):
      %10 = wasmssa.const 2 : i32
      %11 = wasmssa.add %arg1 %10 : i32
      wasmssa.block_return %11 : i32
    }, {
    }) : (i32, i32) -> ()
  ^bb1(%9: i32):  // pred: ^bb0
    wasmssa.block_return %9 : i32
  }, {
    %4 = wasmssa.const 1 : i32
    wasmssa.block_return %4 : i32
  }) : (i32) -> ()
^bb1(%3: i32):  // pred: ^bb0
  wasmssa.return %3 : i32
}
