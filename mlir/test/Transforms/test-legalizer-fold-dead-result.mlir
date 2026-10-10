// RUN: mlir-opt -split-input-file -test-legalize-patterns="allow-pattern-rollback=1" %s | FileCheck %s --check-prefix=ROLLBACK
// RUN: mlir-opt -split-input-file -test-legalize-patterns="allow-pattern-rollback=0" %s | FileCheck %s --check-prefix=NO-ROLLBACK

// Constant materialization fails for result 2. The "no rollback" mode skips
// result 2, because it has no uses, and erases the op. The "rollback" mode
// materializes every result, so the fold fails and the op stays. The remarks
// differ between the modes, so this file does not verify them.

// ROLLBACK-LABEL: func @fold_dead_unmaterializable_result
// ROLLBACK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
// ROLLBACK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.op_fold_unmaterializable"(%[[ARG0]])
// ROLLBACK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1)
// NO-ROLLBACK-LABEL: func @fold_dead_unmaterializable_result
// NO-ROLLBACK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
// NO-ROLLBACK-NEXT: %[[C42:[a-z0-9]+]] = "test.constant"() <{value = 42 : i32}> : () -> i32
// NO-ROLLBACK-NEXT: "test.return"(%[[C42]], %[[ARG0]])
func.func @fold_dead_unmaterializable_result(%arg0: i32) -> (i32, i32) {
  %0:3 = "test.op_fold_unmaterializable"(%arg0) : (i32) -> (i32, i32, i32)
  "test.return"(%0#0, %0#1) : (i32, i32) -> ()
}

// -----

// No result has uses. The "no rollback" mode materializes nothing and erases
// the op.

// ROLLBACK-LABEL: func @fold_all_results_dead
// ROLLBACK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
// ROLLBACK-NEXT: "test.op_fold_unmaterializable"(%[[ARG0]])
// ROLLBACK-NEXT: "test.return"()
// NO-ROLLBACK-LABEL: func @fold_all_results_dead
// NO-ROLLBACK-NEXT: "test.return"()
func.func @fold_all_results_dead(%arg0: i32) {
  %0:3 = "test.op_fold_unmaterializable"(%arg0) : (i32) -> (i32, i32, i32)
  "test.return"() : () -> ()
}
