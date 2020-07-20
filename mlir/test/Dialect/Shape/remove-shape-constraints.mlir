// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -remove-shape-constraints -canonicalize <%s | FileCheck %s --dump-input=fail --check-prefixes=CANON,CHECK-BOTH
// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -remove-shape-constraints <%s | FileCheck %s --dump-input=fail --check-prefixes=REPLACE,CHECK-BOTH

// -----
// Check that cstr_broadcastable is removed.
//
// CHECK-BOTH: func @f
func @f(%arg0 : !shape.shape, %arg1 : !shape.shape) -> index {
  // REPLACE-NEXT: %[[WITNESS:.+]] = shape.const_witness true
  // REPLACE-NOT: shape.cstr_eq
  // REPLACE: shape.assuming %[[WITNESS]]
  // CANON-NEXT: test.source
  // CANON-NEXT: return
  %0 = shape.cstr_broadcastable %arg0, %arg1 : !shape.shape, !shape.shape
  %1 = shape.assuming %0 -> index {
    %2 = "test.source"() : () -> (index)
    shape.assuming_yield %2 : index
  }
  return %1 : index
}

// -----
// Check that cstr_eq is removed.
//
// CHECK-BOTH: func @f
func @f(%arg0 : !shape.shape, %arg1 : !shape.shape) -> index {
  // REPLACE-NEXT: %[[WITNESS:.+]] = shape.const_witness true
  // REPLACE-NOT: shape.cstr_eq
  // REPLACE: shape.assuming %[[WITNESS]]
  // CANON-NEXT: test.source
  // CANON-NEXT: return
  %0 = shape.cstr_eq %arg0, %arg1
  %1 = shape.assuming %0 -> index {
    %2 = "test.source"() : () -> (index)
    shape.assuming_yield %2 : index
  }
  return %1 : index
}

// -----
// With a non-const value, we cannot fold away the code, but all constraints
// should be removed still.
//
// CHECK-BOTH: func @f
func @f(%arg0 : !shape.shape, %arg1 : !shape.shape) -> index {
  // CANON-NEXT: test.source
  // CANON-NEXT: return
  %0 = shape.cstr_broadcastable %arg0, %arg1 : !shape.shape, !shape.shape
  %1 = shape.cstr_eq %arg0, %arg1
  %2 = shape.assuming_all %0, %1
  %3 = shape.assuming %0 -> index {
    %4 = "test.source"() : () -> (index)
    shape.assuming_yield %4 : index
  }
  return %3 : index
}
