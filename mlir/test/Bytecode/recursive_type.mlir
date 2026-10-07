// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s
// RUN: mlir-opt -emit-bytecode %s | llvm-strings | FileCheck %s --check-prefix=NATIVE

// The round-trip alone proves nothing, the assembly fallback round-trips too.
// What must not appear in the blob is the type's printed form, only embedded
// when the fallback is used.
// NATIVE: test
// NATIVE-NOT: test_rec_alias

// A mutable type needs tryStartCyclicRead to have a custom encoding: the body
// of !test.test_rec_alias may name the type being defined, so the type has to
// exist before its body can be read. The reader publishes it first, reads the
// body, then sets it.

// CHECK-DAG: ![[$SELF:[^ ]+]] = !test.test_rec_alias<self, !test.test_rec_alias<self>>
// CHECK-DAG: ![[$C:[^ ]+]] = !test.test_rec_alias<c, !test.test_rec_alias<a, !test.test_rec_alias<b, !test.test_rec_alias<c>>>>
// CHECK-DAG: ![[$NESTED:[^ ]+]] = !test.test_rec_alias<nested, tuple<!test.test_rec_alias<nested>, i32>>
// CHECK-DAG: ![[$PLAIN:[^ ]+]] = !test.test_rec_alias<plain, i32>
// CHECK-DAG: ![[$B:[^ ]+]] = !test.test_rec_alias<b, ![[$C]]>
// CHECK-DAG: ![[$A:[^ ]+]] = !test.test_rec_alias<a, ![[$B]]>
// CHECK-DAG: ![[$X:[^ ]+]] = !test.test_rec_alias<x, tuple<tuple<tuple<tuple<tuple<i32>>>>>>
// CHECK-DAG: ![[$ROOT:[^ ]+]] = !test.test_rec_alias<root, tuple<![[$X]], i32>>

// CHECK-LABEL: @roundtrip
func.func @roundtrip() {
  // A type whose body is itself: the case that recurses forever in a reader
  // with no way to hand back a partially built type.
  // CHECK: ![[$SELF]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<self, !test.test_rec_alias<self>>

  // A cycle spanning three entries rather than one, so breaking it needs more
  // than recognising "this is the entry I am already in".
  // CHECK: ![[$A]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<a, !test.test_rec_alias<b, !test.test_rec_alias<c, !test.test_rec_alias<a>>>>

  // The self-reference reached through another type, so the reference back is
  // not the immediate body.
  // CHECK: ![[$NESTED]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<nested, tuple<!test.test_rec_alias<nested>, i32>>

  // A mutable type nested deep enough to defer: first reached through
  // another entry's body, so only the completion pass parses it.
  // CHECK: ![[$ROOT]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<root, tuple<!test.test_rec_alias<x, tuple<tuple<tuple<tuple<tuple<i32>>>>>>, i32>>

  // Not recursive at all, to keep the ordinary path covered.
  // CHECK: ![[$PLAIN]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<plain, i32>

  // A second use of an already-resolved type.
  // CHECK: ![[$PLAIN]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<plain, i32>

  return
}
