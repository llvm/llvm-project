// RUN: mlir-translate -mlir-to-llvmir %s -split-input-file | FileCheck %s

llvm.func @callee(!llvm.ptr, i32, i32) -> i32

// CHECK: define i32 @test_indirectbr_phi(ptr %[[IN_PTR:.*]], ptr %[[ARG1:.*]], i32 %[[ARG2:.*]]) {
llvm.func @test_indirectbr_phi(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg3: i32) -> i32 {
  %0 = llvm.mlir.undef : i1
  %2 = llvm.blockaddress <function = @test_indirectbr_phi, tag = <id = 1>> : !llvm.ptr
  %3 = llvm.mlir.constant(1 : i32) : i32
  %4 = llvm.mlir.constant(2 : i32) : i32
  %5 = llvm.select %0, %2, %arg0 : i1, !llvm.ptr
  // CHECK:   %[[BA0:.*]] = select i1 undef, ptr blockaddress(@test_indirectbr_phi, %[[RET_BB:.*]]), ptr %[[IN_PTR]]
  // CHECK:   indirectbr ptr %[[BA0]], [label %[[BB1:.*]], label %[[BB2:.*]]]
  llvm.indirectbr %5 : !llvm.ptr, [
  ^bb1,
  ^bb2(%3 : i32)
  ]
^bb1:
  // CHECK: [[BB1]]:
  // CHECK:   %[[BA1:.*]] = select i1 undef, ptr blockaddress(@test_indirectbr_phi, %[[RET_BB]]), ptr %[[IN_PTR]]
  // CHECK:   indirectbr ptr %[[BA1]], [label %[[BB2]], label %[[RET_BB]]]
  %6 = llvm.select %0, %2, %arg0 : i1, !llvm.ptr
  llvm.indirectbr %6 : !llvm.ptr, [
  ^bb2(%4 : i32),
  ^bb3
  ]
^bb2(%7: i32):
  // CHECK: [[BB2]]:
  // CHECK:   %[[PHI:.*]] = phi i32 [ 2, %[[BB1]] ], [ 1, {{.*}} ]
  // CHECK:   %[[CALL:.*]] = call i32 @callee(ptr %[[ARG1]], i32 %[[ARG2]], i32 %[[PHI]])
  // CHECK:   ret i32 %[[CALL]]
  %8 = llvm.call @callee(%arg1, %arg3, %7) : (!llvm.ptr, i32, i32) -> i32
  llvm.return %8 : i32
^bb3:
  // CHECK: [[RET_BB]]:
  // CHECK:   ret i32 %[[ARG2]]
  // CHECK: }
  llvm.blocktag <id = 1>
  llvm.return %arg3 : i32
}
