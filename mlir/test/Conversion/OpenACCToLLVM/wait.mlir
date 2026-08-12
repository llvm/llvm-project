// RUN: mlir-opt %s -acc-to-llvm -split-input-file | FileCheck %s

// Empty wait: waitNum=0, null wait list, sync async queue (-1).
// CHECK-LABEL: llvm.func @test_wait_empty
// CHECK: llvm.mlir.constant(-1 : i64)
// CHECK: llvm.mlir.constant(0 : i32)
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @__tgt_acc_wait

module {
  func.func @test_wait_empty() {
    acc.wait
    return
  }
}

// -----

// Wait operands: waitNum matches the list length, values are stored into an
// alloca'd wait list, and the async queue remains sync (-1).
// CHECK-LABEL: llvm.func @test_wait_operands
// CHECK: %[[ASYNC:.*]] = llvm.mlir.constant(-1 : i64)
// CHECK: %[[WAIT0:.*]] = llvm.mlir.constant(0 : i64)
// CHECK: %[[WAIT_NUM:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[WAIT_LIST:.*]] = llvm.alloca %[[WAIT_NUM]] x i64
// CHECK: %[[IDX:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[WAIT_SLOT:.*]] = llvm.getelementptr %[[WAIT_LIST]][%[[IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK: llvm.store %[[WAIT0]], %[[WAIT_SLOT]] : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_wait(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[WAIT_NUM]], %[[WAIT_LIST]], %[[ASYNC]])

module {
  func.func @test_wait_operands() {
    %c0 = arith.constant 0 : i32
    acc.wait(%c0 : i32)
    return
  }
}

// -----

// Async with an explicit queue value: the operand is widened to i64 (folded
// through ArithToLLVM into a constant here) and passed as the async queue.
// CHECK-LABEL: llvm.func @test_wait_async
// CHECK: llvm.mlir.constant(1 : i64)
// CHECK: llvm.call @__tgt_acc_wait

module {
  func.func @test_wait_async() {
    %c1 = arith.constant 1 : i32
    acc.wait async(%c1 : i32)
    return
  }
}

// -----

// Non-constant async operand: must widen i32 to i64 before the runtime call
// (constant folding hides this on the previous test).
// CHECK-LABEL: llvm.func @test_wait_async_arg
// CHECK: llvm.sext %{{.*}} : i32 to i64
// CHECK: llvm.call @__tgt_acc_wait

module {
  func.func @test_wait_async_arg(%q: i32) {
    acc.wait async(%q : i32)
    return
  }
}

// -----

// Async with no value: OpenACC async sentinel -4.
// CHECK-LABEL: llvm.func @test_wait_async_noval
// CHECK: llvm.mlir.constant(-4 : i64)
// CHECK: llvm.call @__tgt_acc_wait

module {
  func.func @test_wait_async_noval() {
    acc.wait async
    return
  }
}

// -----

// CHECK-LABEL: llvm.func @test_wait_if
// CHECK: llvm.cond_br %{{.*}}, ^[[THEN:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: llvm.call @__tgt_acc_wait
// CHECK: llvm.br ^[[CONT]]
// CHECK: ^[[CONT]]:

module {
  func.func @test_wait_if(%cond: i1) {
    acc.wait if(%cond)
    return
  }
}

// -----

// The ident is a constant global whose source field points at the location
// string. Call sites take the address of that ident, not of the string.

// CHECK: llvm.mlir.global internal constant @[[$SRC:loc_5_3_[0-9]+]](";wait.mlir;test_wait_with_loc;5;3;;\00")
// CHECK: llvm.mlir.global internal constant @[[$IDENT:ident_loc_5_3_[0-9]+]]() {{.*}} : !llvm.struct<(i32, i32, i32, i32, ptr)> {
// CHECK: llvm.mlir.zero : !llvm.struct<(i32, i32, i32, i32, ptr)>
// CHECK: llvm.mlir.addressof @[[$SRC]]
// CHECK: llvm.getelementptr
// CHECK: llvm.insertvalue {{.*}}[4]
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @test_wait_with_loc
// CHECK: llvm.mlir.addressof @[[$IDENT]]
// CHECK: llvm.call @__tgt_acc_wait

#loc = loc("wait.mlir":5:3)
module {
  func.func @test_wait_with_loc() {
    acc.wait loc(#loc)
    return
  }
}

// -----

// Operations without file:line information fall back to an unknown ident.

// CHECK: llvm.mlir.global internal constant @loc__(";unknown;unknown;0;0;;\00")
// CHECK: llvm.mlir.global internal constant @ident_loc__() {{.*}} : !llvm.struct<(i32, i32, i32, i32, ptr)> {
// CHECK: llvm.mlir.zero : !llvm.struct<(i32, i32, i32, i32, ptr)>
// CHECK: llvm.mlir.addressof @loc__
// CHECK: llvm.getelementptr
// CHECK: llvm.insertvalue {{.*}}[4]
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @test_wait_unknown_loc
// CHECK: llvm.mlir.addressof @ident_loc__
// CHECK: llvm.call @__tgt_acc_wait

module {
  func.func @test_wait_unknown_loc() {
    acc.wait loc(unknown)
    return
  }
}
