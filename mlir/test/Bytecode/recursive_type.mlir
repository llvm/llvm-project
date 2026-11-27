// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s
// RUN: mlir-opt -emit-bytecode %s | llvm-strings | FileCheck %s --check-prefix=NATIVE --implicit-check-not=test_rec_alias

// The round-trip alone proves nothing, the assembly fallback round-trips too.
// What must not appear in the blob is the type's printed form, only embedded
// when the fallback is used.
// NATIVE: test

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
// CHECK-DAG: ![[$D:[^ ]+]] = !test.test_rec_alias<d, tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<d>>>>>>>>
// CHECK-DAG: ![[$DATTR:[^ ]+]] = !test.test_rec_alias<dattr, tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<dattr>>>>>>>>
// CHECK-DAG: ![[$DFUNC:[^ ]+]] = !test.test_rec_alias<dfunc, tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<dfunc>>>>>>>>
// CHECK-DAG: ![[$WRAPPED:[^ ]+]] = !test.test_rec_alias<wrapped, tuple<!test.test_rec_alias<wrapped>>>
// CHECK-DAG: ![[$DEEP_WRAPPED:[^ ]+]] = !test.test_rec_alias<deep_wrapped, tuple<tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<deep_wrapped>>>>>>>>>
// Each alias in c9..c0 is emitted before its user, while the chain as a whole
// may interleave with unrelated top-level aliases above.
// CHECK-DAG: ![[$C9:[^ ]+]] = !test.test_rec_alias<c9, !test.test_rec_alias<c0, !test.test_rec_alias<c1, !test.test_rec_alias<c2, !test.test_rec_alias<c3, !test.test_rec_alias<c4, !test.test_rec_alias<c5, !test.test_rec_alias<c6, !test.test_rec_alias<c7, !test.test_rec_alias<c8, !test.test_rec_alias<c9>>>>>>>>>>>
// CHECK-DAG: ![[$C8:[^ ]+]] = !test.test_rec_alias<c8, ![[$C9]]>
// CHECK-DAG: ![[$C7:[^ ]+]] = !test.test_rec_alias<c7, ![[$C8]]>
// CHECK-DAG: ![[$C6:[^ ]+]] = !test.test_rec_alias<c6, ![[$C7]]>
// CHECK-DAG: ![[$C5:[^ ]+]] = !test.test_rec_alias<c5, ![[$C6]]>
// CHECK-DAG: ![[$C4:[^ ]+]] = !test.test_rec_alias<c4, ![[$C5]]>
// CHECK-DAG: ![[$C3:[^ ]+]] = !test.test_rec_alias<c3, ![[$C4]]>
// CHECK-DAG: ![[$C2:[^ ]+]] = !test.test_rec_alias<c2, ![[$C3]]>
// CHECK-DAG: ![[$C1:[^ ]+]] = !test.test_rec_alias<c1, ![[$C2]]>
// CHECK-DAG: ![[$C0:[^ ]+]] = !test.test_rec_alias<c0, ![[$C1]]>

// CHECK-LABEL: @roundtrip
func.func @roundtrip() {
  // A type whose body is itself: the case that recurses forever in a reader
  // with no way to hand back a partially built type.
  // CHECK: ![[$SELF]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<self, !test.test_rec_alias<self>>

  // A cycle spanning three entries rather than one, so breaking it needs more
  // than recognizing "this is the entry I am already in".
  // CHECK: ![[$A]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<a, !test.test_rec_alias<b, !test.test_rec_alias<c, !test.test_rec_alias<a>>>>

  // The self-reference reached through another type, so the reference back is
  // not the immediate body.
  // CHECK: ![[$NESTED]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<nested, tuple<!test.test_rec_alias<nested>, i32>>

  // A mutable type nested deep enough to defer: first reached through
  // another entry's body, so the worklist resolves it before its parent.
  // CHECK: ![[$ROOT]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<root, tuple<!test.test_rec_alias<x, tuple<tuple<tuple<tuple<tuple<i32>>>>>>, i32>>

  // Not recursive at all, to keep the ordinary path covered.
  // CHECK: ![[$PLAIN]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<plain, i32>

  // A second use of an already-resolved type.
  // CHECK: ![[$PLAIN]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<plain, i32>

  // A self-reference through six tuples: one past the depth limit, so the
  // reference back is read through the depth branch, not the nested path.
  // CHECK: ![[$D]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<d, tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<d>>>>>>>>

  // The same record first reached from an attribute, through five
  // enclosing tuples, so the back-reference lands past the depth
  // limit on the attribute path too.
  // CHECK: test_attr = tuple<tuple<tuple<tuple<tuple<![[$DATTR]]>>>>>
  "test.dummy_op_for_roundtrip"() {test_attr = tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<dattr, tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<dattr>>>>>>>>>>>>>} : () -> ()

  // A cycle spanning ten distinct mutable types, exceeding maxAttrTypeDepth so
  // later mutable types in the cycle are first discovered via deferral.
  // CHECK: ![[$C0]]
  "test.dummy_op_for_roundtrip"() : () -> !test.test_rec_alias<c0, !test.test_rec_alias<c1, !test.test_rec_alias<c2, !test.test_rec_alias<c3, !test.test_rec_alias<c4, !test.test_rec_alias<c5, !test.test_rec_alias<c6, !test.test_rec_alias<c7, !test.test_rec_alias<c8, !test.test_rec_alias<c9, !test.test_rec_alias<c0>>>>>>>>>>>

  // A mutable type first reached through the immutable type wrapping it in its
  // own body: reading its body re-enters the active tuple<...> entry, which is
  // guarded by the published partial.
  // CHECK: tuple<![[$WRAPPED]]>
  "test.dummy_op_for_roundtrip"() : () -> tuple<!test.test_rec_alias<wrapped, tuple<!test.test_rec_alias<wrapped>>>>

  // Same pattern wrapped past maxAttrTypeDepth so deferral and guarded
  // re-entry compose cleanly.
  // CHECK: tuple<tuple<tuple<tuple<tuple<tuple<tuple<![[$DEEP_WRAPPED]]>>>>>>>
  "test.dummy_op_for_roundtrip"() : () -> tuple<tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<deep_wrapped, tuple<tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<deep_wrapped>>>>>>>>>>>>>>>>

  return
}

// The same record first reached from a function type, through five
// enclosing tuples, so the back-reference lands past the depth limit
// on the function-type path too.
// CHECK: func.func private @from_function_type(tuple<tuple<tuple<tuple<tuple<![[$DFUNC]]>>>>>)
func.func private @from_function_type(tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<dfunc, tuple<tuple<tuple<tuple<tuple<tuple<!test.test_rec_alias<dfunc>>>>>>>>>>>>>)
