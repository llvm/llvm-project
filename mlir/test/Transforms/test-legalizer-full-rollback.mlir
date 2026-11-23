// RUN: mlir-opt -allow-unregistered-dialect -test-legalize-patterns="test-legalize-mode=full" -split-input-file -verify-diagnostics %s | FileCheck %s

// Test that region inlining can be properly undone.

// CHECK-LABEL: func @test_undo_region_inline() {
//       CHECK:    "test.region"() ({
//       CHECK:    ^{{.*}}(%[[arg0:.*]]: i64):
//       CHECK:      cf.br ^[[bb1:.*]](%[[arg0]] : i64)
//       CHECK:    ^[[bb1]](%[[arg1:.*]]: i64):
//       CHECK:      "test.invalid"(%[[arg1]]) : (i64) -> ()
//       CHECK:    }) : () -> ()
//       CHECK:    "test.return"() : () -> ()
//       CHECK: }

// expected-remark@+1 {{applyFullConversion failed}}
builtin.module {
func.func @test_undo_region_inline() {
  "test.region"() ({
    ^bb1(%i0: i64):
      // expected-error@+1 {{failed to legalize operation 'cf.br'}}
      cf.br ^bb2(%i0 : i64)
    ^bb2(%i1: i64):
      "test.invalid"(%i1) : (i64) -> ()
  }) {} : () -> ()

  "test.return"() : () -> ()
}
}

// -----

// Test that multiple block erases can be properly undone.

// CHECK-LABEL: func @test_undo_block_erase() {
//       CHECK:   "test.region"() ({
//       CHECK:   ^{{.*}}(%[[arg0:.*]]: i64):
//       CHECK:     cf.br ^[[bb2:.*]](%[[arg0]] : i64)
//       CHECK:   ^[[bb1:.*]](%[[arg1:.*]]: i64):
//       CHECK:     "test.invalid"(%[[arg1]]) : (i64) -> ()
//       CHECK:   ^[[bb2]](%[[arg2:.*]]: i64):
//       CHECK:     cf.br ^[[bb1]](%[[arg2]] : i64)
//       CHECK:   }) {legalizer.erase_old_blocks, legalizer.should_clone} : () -> ()
//       CHECK:   "test.return"() : () -> ()
//       CHECK: }

// expected-remark@+1 {{applyFullConversion failed}}
builtin.module {
func.func @test_undo_block_erase() {
  // expected-error@+1 {{failed to legalize operation 'test.region'}}
  "test.region"() ({
    ^bb1(%i0: i64):
      cf.br ^bb3(%i0 : i64)
    ^bb2(%i1: i64):
      "test.invalid"(%i1) : (i64) -> ()
    ^bb3(%i2: i64):
      cf.br ^bb2(%i2 : i64)
  }) {legalizer.should_clone, legalizer.erase_old_blocks} : () -> ()
  "test.return"() : () -> ()
}
}
