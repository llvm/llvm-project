// RUN: mlir-opt %s -test-print-callgraph -split-input-file -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "simple"
module attributes {test.name = "simple"} {

  // CHECK: Node{{.*}}func_a
  func.func @func_a() {
    return
  }

  // CHECK-NOT: Node{{.*}}func_b
  func.func private @func_b()

  // CHECK: Node{{.*}}func_c{{.*}}private
  // CHECK-NEXT: Call-Edge{{.*}}Unknown-Callee-Node
  func.func private @func_c() {
    call @func_b() : () -> ()
    return
  }

  // CHECK: Node{{.*}}func_d
  // CHECK-NEXT: Call-Edge{{.*}}func_c{{.*}}private
  func.func @func_d() {
    call @func_c() : () -> ()
    return
  }

  // CHECK: Node{{.*}}func_e
  // CHECK-DAG: Call-Edge{{.*}}func_c{{.*}}private
  // CHECK-DAG: Call-Edge{{.*}}func_d
  // CHECK-DAG: Call-Edge{{.*}}func_e
  // CHECK-DAG: Ref-Edge{{.*}}func_a
  func.func @func_e() {
    call @func_c() : () -> ()
    call @func_d() : () -> ()
    call @func_e() { use = @func_a } : () -> ()
    return
  }

  // CHECK: Node{{.*}}func_f
  // CHECK: Child-Edge{{.*}}test.functional_region_op
  // CHECK: Call-Edge{{.*}}test.functional_region_op
  func.func @func_f() {
    // CHECK: Node{{.*}}test.functional_region_op
    // CHECK: Call-Edge{{.*}}func_f
    %fn = "test.functional_region_op"() ({
      call @func_f() : () -> ()
      "test.return"() : () -> ()
    }) : () -> (() -> ())

    call_indirect %fn() : () -> ()
    return
  }

  // CHECK: Node{{.*}}func_g
  // CHECK: Ref-Edge{{.*}}func_c
  // CHECK: Call-Edge{{.*}}Unknown-Callee-Node
  func.func @func_g() -> (() -> ()) {
    // A private symbol maybe escaped.
    %0 = func.constant @func_c : () -> ()
    call_indirect %0() : () -> ()
    return %0 : () -> ()
  }

  // CHECK: Node{{.*}}func_h{{.*}}private
  func.func private @func_h() {
    return
  }

  // non-callable top level operation reference a non-symbolic callable node.
  %0 = "test.functional_region_op"() ({
    func.call @func_f() : () -> ()
    "test.return"() : () -> ()
  }) : () -> (() -> ())

  // If a referenced symbol only has declaration, there would not be
  // a reference edge from external node to it.
  "live.user"() { use = @func_b } : () -> ()

  // non-callable top level operation reference callable symbol.
  "live.user"() { use = @func_c } : () -> ()
  func.call @func_h() : () -> ()

  // CHECK: Node{{.*}}External-Caller-Node
  // CHECK-NEXT: Ref-Edge{{.*}}func_a
  // CHECK-NEXT: Ref-Edge{{.*}}func_d
  // CHECK-NEXT: Ref-Edge{{.*}}func_e
  // CHECK-NEXT: Ref-Edge{{.*}}func_f
  // CHECK-NEXT: Ref-Edge{{.*}}func_g
  // CHECK-NEXT: Ref-Edge{{.*}}test.functional_region_op
  // CHECK-NOT: Ref-Edge{{.*}}func_b
  // CHECK-NEXT: Ref-Edge{{.*}}func_c{{.*}}private
  // CHECK-NEXT: Ref-Edge{{.*}}func_h{{.*}}private

  // CHECK: Node{{.*}}Unknown-Callee-Node
}

// -----

// CHECK-LABEL: Testing : "nested"
module attributes {test.name = "nested"} {
  module @nested_module {
    // CHECK: Node{{.*}}func_a{{.*}}nested
    func.func nested @func_a() {
      return
    }
    // CHECK: Node{{.*}}func_b{{.*}}nested
    func.func nested @func_b() {
      return
    }
  }

  // CHECK: Node{{.*}}func_c
  // CHECK: Call-Edge{{.*}}func_a{{.*}}nested
  // CHECK: Ref-Edge{{.*}}func_b{{.*}}nested
  func.func @func_c() {
    "test.conversion_call_op"() { use = @nested_module::@func_b, callee = @nested_module::@func_a } : () -> ()
    return
  }

  // CHECK: Node{{.*}}External-Caller-Node
  // CHECK-NEXT: Ref-Edge{{.*}}func_c
  // CHECK-NOT: Ref-Edge{{.*}}func_a
  // CHECK-NOT: Ref-Edge{{.*}}func_b

  // CHECK: Node{{.*}}Unknown-Callee-Node
}

// -----

// CHECK-LABEL: Testing : "SCC"
// CHECK: SCCs
module attributes {test.name = "SCC"} {
  // CHECK: SCC :
  // CHECK-NEXT: Node{{.*}}Unknown-Callee-Node

  // CHECK: SCC :
  // CHECK-NEXT: Node{{.*}}foo
  func.func @foo(%arg0 : () -> ()) {
    call_indirect %arg0() : () -> ()
    return
  }

  // CHECK: SCC :
  // CHECK-NEXT: Node{{.*}}bar
  func.func @bar(%arg1 : () -> ()) {
    call_indirect %arg1() : () -> ()
    return
  }

  // CHECK: SCC :
  // CHECK-NEXT: Node{{.*}}External-Caller-Node
}
