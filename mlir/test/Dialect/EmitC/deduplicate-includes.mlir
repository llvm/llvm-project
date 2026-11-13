// RUN: mlir-opt %s -split-input-file -deduplicate-emitc-includes | FileCheck %s
// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(deduplicate-emitc-includes,deduplicate-emitc-includes)' | FileCheck %s

// Modules without includes are left unchanged.
// CHECK-LABEL: module @no_headers {
// CHECK-NEXT: }
module @no_headers {
}

// -----

// Unique headers keep their original order.
// CHECK-LABEL: module @existing_headers {
// CHECK-NEXT: emitc.include <"string">
// CHECK-NEXT: emitc.include <"map">
// CHECK-NEXT: }
module @existing_headers {
  emitc.include <"string">
  emitc.include <"map">
}

// -----

// Remove only repeated headers, retaining the first occurrence in place.
// CHECK-LABEL: module @duplicate_headers {
// CHECK-NEXT: emitc.include <"string">
// CHECK-NEXT: emitc.include <"map">
// CHECK-NEXT: emitc.include <"math.h">
// CHECK-NEXT: }
module @duplicate_headers {
  emitc.include <"string">
  emitc.include <"map">
  emitc.include <"string">
  emitc.include <"string">
  emitc.include <"math.h">
  emitc.include <"map">
}

// -----

// Quoted includes remain unchanged, including duplicates and names that match
// standard includes. They do not suppress the first standard include.
// CHECK-LABEL: module @quoted_headers {
// CHECK-NEXT: emitc.include "map"
// CHECK-NEXT: emitc.include <"map">
// CHECK-NEXT: emitc.include "map"
// CHECK-NEXT: }
module @quoted_headers {
  emitc.include "map"
  emitc.include <"map">
  emitc.include "map"
  emitc.include <"map">
}

// -----

// Nested modules share header tracking with their enclosing module. The first
// occurrence can be nested; cleanup does not move it to the outer module.
// CHECK-LABEL: module @nested_headers {
// CHECK-NEXT: emitc.include <"map">
// CHECK-NEXT: module @first {
// CHECK-NEXT: emitc.include <"string">
// CHECK-NEXT: }
// CHECK-NEXT: module @second {
// CHECK-NEXT: emitc.include <"math.h">
// CHECK-NEXT: }
// CHECK-NEXT: emitc.func @method() {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module @nested_headers {
  emitc.include <"map">
  module @first {
    emitc.include <"string">
    emitc.include <"map">
  }
  emitc.include <"string">
  module @second {
    emitc.include <"map">
    emitc.include <"math.h">
    emitc.include <"math.h">
  }
  emitc.func @method() {
    emitc.include <"map">
    emitc.include <"string">
    emitc.return
  }
  emitc.include <"math.h">
}

// -----

// Each file is independent of other files and of the enclosing module. File
// headers must not suppress later outer-module headers or reset outer tracking.
// CHECK-LABEL: module @file_headers {
// CHECK-NEXT: emitc.include <"map">
// CHECK-NEXT: emitc.file "one" {
// Inside emitc.file the printer omits "emitc.".
// CHECK-NEXT: include <"map">
// CHECK-NEXT: include <"stdlib.h">
// CHECK-NEXT: }
// CHECK-NEXT: emitc.file "two" {
// CHECK-NEXT: include <"map">
// CHECK-NEXT: }
// CHECK-NEXT: emitc.include <"stdlib.h">
// CHECK-NEXT: }
module @file_headers {
  emitc.include <"map">
  emitc.file "one" {
    emitc.include <"map">
    emitc.include <"map">
    emitc.include <"stdlib.h">
  }
  emitc.file "two" {
    emitc.include <"map">
    emitc.include <"map">
  }
  emitc.include <"stdlib.h">
  emitc.include <"stdlib.h">
  emitc.include <"map">
}
