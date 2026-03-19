// RUN: mlir-opt %s -acc-bind-routine -split-input-file | FileCheck %s

// Call to routine with bind is rewritten to the bound symbol inside
// offload region.
module {
  acc.routine @r_bind func(@foo) seq bind(@bar)
  func.func @foo() attributes {acc.routine_info = #acc.routine_info<[@r_bind]>} {
    return
  }
  func.func @bar() {
    return
  }
  func.func @main() {
    acc.serial {
      func.call @foo() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @r_bind func(@foo) bind(@bar) seq
// CHECK: func.call @bar() : () -> ()

// -----

// Bind with string name: call is rewritten to the string symbol.
module {
  acc.routine @r_bind_str func(@wrapped) seq bind("actual_impl")
  func.func @wrapped() attributes {acc.routine_info = #acc.routine_info<[@r_bind_str]>} {
    return
  }
  func.func @actual_impl() {
    return
  }
  func.func @entry() {
    acc.serial {
      func.call @wrapped() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: func.call @actual_impl() : () -> ()

// -----

// Call outside offload region is unchanged.
module {
  acc.routine @r_bind func(@foo) seq bind(@bar)
  func.func @foo() attributes {acc.routine_info = #acc.routine_info<[@r_bind]>} {
    return
  }
  func.func @bar() {
    return
  }
  func.func @main() {
    func.call @foo() : () -> ()
    acc.serial {
      func.call @foo() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: call @foo() : () -> ()
// CHECK: func.call @bar() : () -> ()

// -----

// Indirect call (callee is value) is skipped; no crash.
module {
  acc.routine @r_bind func(@target) seq bind(@bound)
  func.func @target() attributes {acc.routine_info = #acc.routine_info<[@r_bind]>} {
    return
  }
  func.func @bound() {
    return
  }
  func.func @caller(%callee: () -> ()) {
    acc.serial {
      func.call_indirect %callee() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: func.call_indirect %{{.*}}() : () -> ()
