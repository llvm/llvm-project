// RUN: mlir-opt -test-target-materialization-with-no-uses %s | FileCheck %s

// The conversion is set up as follows:
// - type_changer ops are illegal;
// - type_changer ops are replaced with their operands;
// - i16 types are converted to i64 by the type conversion;
// - the rest of the types are legal.
// The first type_changer is replaced with its operand. For the pattern to
// apply to the second type_changer, the conversion infra creates a dummy
// cast operation to cast from the i32 to i64 because the original op takes an
// (illegal) i16 that became i64. This dummy operation should be replaced by
// the one produced by the target materialization hook. At the moment when the
// materialization decision is taken, the i64 replacement of the first type
// change (the result of the dummy cast) has no uses, but the value it replaces
// does, so the infra must call the materialization rather than assume the
// dummy cast to be dead.

// CHECK-LABEL: @foo
func.func @foo() {
  %0 = "test.type_producer"() : () -> i32
  // CHECK: test.cast
  // CHECK-NOT: test.type_changer
  %1 = "test.type_changer"(%0) : (i32) -> i16
  %2 = "test.type_changer"(%1) : (i16) -> i64
  "test.type_consumer"(%2) : (i64) -> ()
  return
}
