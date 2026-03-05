// RUN: mlir-opt %s -test-pdl-bytecode-pass -split-input-file --debug 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test that named PDL patterns have their debug name set.
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@named_pattern(%root : !pdl.operation) : benefit(1), loc([%root]), root("test.op") -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @named_pattern(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.replaced_by_pattern"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK: Pattern named_pattern
module @ir {
  "test.op"() : () -> ()
}