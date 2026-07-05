// RUN: mlir-opt -split-input-file --verify-roundtrip %s | FileCheck %s

llvm.func @ib0(%dest : !llvm.ptr, %arg0 : i32, %arg1 : i32) -> i32 {
  llvm.indirectbr %dest : !llvm.ptr, [
  ^head(%arg0 : i32),
  ^tail(%arg1, %arg0 : i32, i32)
  ]
^head(%r0 : i32):
  llvm.return %r0 : i32
^tail(%r1 : i32, %r2 : i32):
  %r = llvm.add %r1, %r2 : i32
  llvm.return %r : i32
}

// CHECK: llvm.func @ib0(%[[Addr:.*]]: !llvm.ptr, %[[A0:.*]]: i32, %[[A1:.*]]: i32) -> i32 {
// CHECK:   llvm.indirectbr %[[Addr]] : !llvm.ptr, [
// CHECK:   ^bb1(%[[A0:.*]] : i32)
// CHECK:   ^bb2(%[[A1:.*]], %[[A0]] : i32, i32)
// CHECK:   ]
// CHECK: ^bb1(%[[Op0:.*]]: i32):
// CHECK:   llvm.return %[[Op0]] : i32
// CHECK: ^bb2(%[[Op1:.*]]: i32, %[[Op2:.*]]: i32):
// CHECK:   %[[Op3:.*]] = llvm.add %[[Op1]], %[[Op2]] : i32
// CHECK:   llvm.return %[[Op3]] : i32
// CHECK: }

// -----

llvm.func @ib1(%dest : !llvm.ptr) {
  llvm.indirectbr %dest : !llvm.ptr, []
}

// CHECK: llvm.func @ib1(%[[Addr:.*]]: !llvm.ptr) {
// CHECK:   llvm.indirectbr %[[Addr]] : !llvm.ptr, []
// CHECK: }

// -----

// CHECK: llvm.func @test_indirectbr_phi(
// CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> i32 {
llvm.func @callee(!llvm.ptr, i32, i32) -> i32
llvm.func @test_indirectbr_phi(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> i32 {
  %0 = llvm.mlir.undef : i1
  %1 = llvm.mlir.addressof @test_indirectbr_phi : !llvm.ptr
  %2 = llvm.blockaddress <function = @test_indirectbr_phi, tag = <id = 1>> : !llvm.ptr
  // CHECK:   %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.mlir.constant(1 : i32) : i32
  // CHECK:   %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
  %4 = llvm.mlir.constant(2 : i32) : i32
  %5 = llvm.select %0, %2, %arg0 : i1, !llvm.ptr
  // CHECK:   llvm.indirectbr {{.*}} : !llvm.ptr, [
  // CHECK:   ^[[HEAD_BB:.*]],
  // CHECK:   ^[[TAIL_BB:.*]](%[[ONE]] : i32)
  // CHECK:   ]
  llvm.indirectbr %5 : !llvm.ptr, [
  ^bb1,
  ^bb2(%3 : i32)
  ]
^bb1:
  // CHECK: ^[[HEAD_BB]]:
  // CHECK:   llvm.indirectbr {{.*}} : !llvm.ptr, [
  // CHECK:   ^[[TAIL_BB]](%[[TWO]] : i32),
  // CHECK:   ^[[END_BB:.*]]
  // CHECK:   ]
  %6 = llvm.select %0, %2, %arg0 : i1, !llvm.ptr
  llvm.indirectbr %6 : !llvm.ptr, [
  ^bb2(%4 : i32),
  ^bb3
  ]
^bb2(%7: i32):
  // CHECK: ^[[TAIL_BB]](%[[BLOCK_ARG:.*]]: i32):
  // CHECK:   {{.*}} = llvm.call @callee({{.*}}, %[[BLOCK_ARG]])
  // CHECK:   llvm.return
  %8 = llvm.call @callee(%arg1, %arg3, %7) : (!llvm.ptr, i32, i32) -> i32
  llvm.return %8 : i32
^bb3:
  // CHECK: ^[[END_BB]]:
  // CHECK:   llvm.blocktag
  // CHECK:   llvm.return
  // CHECK: }
  llvm.blocktag <id = 1>
  llvm.return %arg3 : i32
}
