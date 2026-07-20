// RUN: mlir-opt %s -lower-nested-module-to-class -split-input-file | FileCheck %s
// RUN: mlir-opt %s -lower-nested-module-to-class="lower-all=true" -split-input-file | FileCheck %s --check-prefix=ALL

// Case 1: Nested module with class tag.
module @outer1 {
  module @tag_class attributes {emitc.class} {
    emitc.func @foo() {
      emitc.return
    }
  }
}

// CHECK:      module @outer1 {
// CHECK-NEXT:   emitc.class @tag_class {
// CHECK-NEXT:     emitc.func @foo() {
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// ALL:      module @outer1 {
// ALL-NEXT:   emitc.class @tag_class {
// ALL-NEXT:     emitc.func @foo() {
// ALL-NEXT:       return
// ALL-NEXT:     }
// ALL-NEXT:   }
// ALL-NEXT: }

// -----

// Case 2: Nested module matching the heuristic (globals used by functions).
module @outer2 {
  module @heuristic_class {
    emitc.global static const @global_var : !emitc.array<1xi8> = dense<0>
    emitc.func @foo() {
      %0 = emitc.get_global @global_var : !emitc.array<1xi8>
      emitc.return
    }
  }
}

// CHECK:      module @outer2 {
// CHECK-NEXT:   emitc.class @heuristic_class {
// CHECK-NEXT:     emitc.field @global_var : !emitc.array<1xi8> = dense<0>
// CHECK-NEXT:     emitc.func @foo() {
// CHECK-NEXT:       %0 = get_field @global_var : !emitc.array<1xi8>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// Case 3: Nested module that does not match the heuristic, but tag/lower-all is false.
module @outer3 {
  module @helper_module {
    emitc.func @foo() {
      emitc.return
    }
  }
}

// CHECK:      module @outer3 {
// CHECK-NEXT:   module @helper_module {
// CHECK-NEXT:     emitc.func @foo() {
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// ALL:      module @outer3 {
// ALL-NEXT:   emitc.class @helper_module {
// ALL-NEXT:     emitc.func @foo() {
// ALL-NEXT:       return
// ALL-NEXT:     }
// ALL-NEXT:   }
// ALL-NEXT: }
