// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-legalize-patterns="allow-pattern-rollback=0" -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @conditional_replacement(
//  CHECK-SAME:     %[[arg0:.*]]: i43)
//       CHECK:   %[[cast1:.*]] = "test.cast"(%[[arg0]]) : (i43) -> i42
//       CHECK:   %[[legal:.*]] = "test.legal_op"() : () -> i42
//       CHECK:   %[[cast2:.*]] = "test.cast"(%[[legal]], %[[legal]]) : (i42, i42) -> i42
// Uses were replaced for dummy_user_1.
//       CHECK:   "test.dummy_user_1"(%[[cast2]]) {replace_uses} : (i42) -> ()
// Uses were also replaced for dummy_user_2, but not by value_replace. The uses
// were replaced due to the block signature conversion.
//       CHECK:   "test.dummy_user_2"(%[[cast1]]) : (i42) -> ()
//       CHECK:   "test.value_replace"(%[[cast1]], %[[legal]]) {conditional, is_legal} : (i42, i42) -> ()
func.func @conditional_replacement(%arg0: i42) {
  %repl = "test.legal_op"() : () -> (i42)
  // expected-remark @+1 {{is not legalizable}}
  "test.dummy_user_1"(%arg0) {replace_uses} : (i42) -> ()
  // expected-remark @+1 {{is not legalizable}}
  "test.dummy_user_2"(%arg0) {} : (i42) -> ()
  // Perform a conditional 1:N replacement.
  "test.value_replace"(%arg0, %repl) {conditional} : (i42, i42) -> ()
  "test.return"() : () -> ()
}
