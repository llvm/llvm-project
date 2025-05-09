// RUN: mlir-opt %s --verify-roundtrip | FileCheck %s

// CHECK-LABEL: func @types
// CHECK-SAME:  (%{{.*}}: !smt.bool, %{{.*}}: !smt.bv<32>, %{{.*}}: !smt.int, %{{.*}}: !smt.sort<"uninterpreted_sort">, %{{.*}}: !smt.sort<"uninterpreted_sort"[!smt.bool, !smt.int]>, %{{.*}}: !smt.func<(!smt.bool, !smt.bool) !smt.bool>)
func.func @types(%arg0: !smt.bool, %arg1: !smt.bv<32>, %arg2: !smt.int, %arg3: !smt.sort<"uninterpreted_sort">, %arg4: !smt.sort<"uninterpreted_sort"[!smt.bool, !smt.int]>, %arg5: !smt.func<(!smt.bool, !smt.bool) !smt.bool>) {
  return
}

func.func @core(%in: i8) {
  // CHECK: %a = smt.declare_fun "a" {smt.some_attr} : !smt.bool
  %a = smt.declare_fun "a" {smt.some_attr} : !smt.bool
  // CHECK: smt.declare_fun {smt.some_attr} : !smt.bv<32>
  %b = smt.declare_fun {smt.some_attr} : !smt.bv<32>
  // CHECK: smt.declare_fun {smt.some_attr} : !smt.int
  %c = smt.declare_fun {smt.some_attr} : !smt.int
  // CHECK: smt.declare_fun {smt.some_attr} : !smt.sort<"uninterpreted_sort">
  %d = smt.declare_fun {smt.some_attr} : !smt.sort<"uninterpreted_sort">
  // CHECK: smt.declare_fun {smt.some_attr} : !smt.func<(!smt.int, !smt.bool) !smt.bool>
  %e = smt.declare_fun {smt.some_attr} : !smt.func<(!smt.int, !smt.bool) !smt.bool>

  // CHECK: smt.constant true {smt.some_attr}
  %true = smt.constant true {smt.some_attr}
  // CHECK: smt.constant false {smt.some_attr}
  %false = smt.constant false {smt.some_attr}

  // CHECK: smt.assert %a {smt.some_attr}
  smt.assert %a {smt.some_attr}

  // CHECK: smt.reset {smt.some_attr}
  smt.reset {smt.some_attr}

  // CHECK: smt.push 1 {smt.some_attr}
  smt.push 1 {smt.some_attr}

  // CHECK: smt.pop 1 {smt.some_attr}
  smt.pop 1 {smt.some_attr}

  // CHECK: %{{.*}} = smt.solver(%{{.*}}) {smt.some_attr} : (i8) -> (i8, i32) {
  // CHECK: ^bb0(%{{.*}}: i8)
  // CHECK:   %{{.*}} = smt.check {smt.some_attr} sat {
  // CHECK:     smt.yield %{{.*}} : i32
  // CHECK:   } unknown {
  // CHECK:     smt.yield %{{.*}} : i32
  // CHECK:   } unsat {
  // CHECK:     smt.yield %{{.*}} : i32
  // CHECK:   } -> i32
  // CHECK:   smt.yield %{{.*}}, %{{.*}} : i8, i32
  // CHECK: }
  %0:2 = smt.solver(%in) {smt.some_attr} : (i8) -> (i8, i32) {
  ^bb0(%arg0: i8):
    %1 = smt.check {smt.some_attr} sat {
      %c1_i32 = arith.constant 1 : i32
      smt.yield %c1_i32 : i32
    } unknown {
      %c0_i32 = arith.constant 0 : i32
      smt.yield %c0_i32 : i32
    } unsat {
      %c-1_i32 = arith.constant -1 : i32
      smt.yield %c-1_i32 : i32
    } -> i32
    smt.yield %arg0, %1 : i8, i32
  }

  // CHECK: smt.solver() : () -> () {
  // CHECK-NEXT: }
  smt.solver() : () -> () { }

  // CHECK: smt.solver() : () -> () {
  // CHECK-NEXT: smt.set_logic "AUFLIA"
  // CHECK-NEXT: }
  smt.solver() : () -> () {
    smt.set_logic "AUFLIA"
  }

  //      CHECK: smt.check sat {
  // CHECK-NEXT: } unknown {
  // CHECK-NEXT: } unsat {
  // CHECK-NEXT: }
  smt.check sat { } unknown { } unsat { }

  // CHECK: %{{.*}} = smt.eq %{{.*}}, %{{.*}} {smt.some_attr} : !smt.bv<32>
  %1 = smt.eq %b, %b {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.distinct %{{.*}}, %{{.*}} {smt.some_attr} : !smt.bv<32>
  %2 = smt.distinct %b, %b {smt.some_attr} : !smt.bv<32>

  // CHECK: %{{.*}} = smt.eq %{{.*}}, %{{.*}}, %{{.*}} : !smt.bool
  %3 = smt.eq %a, %a, %a : !smt.bool
  // CHECK: %{{.*}} = smt.distinct %{{.*}}, %{{.*}}, %{{.*}} : !smt.bool
  %4 = smt.distinct %a, %a, %a : !smt.bool

  // CHECK: %{{.*}} = smt.ite %{{.*}}, %{{.*}}, %{{.*}} {smt.some_attr} : !smt.bv<32>
  %5 = smt.ite %a, %b, %b {smt.some_attr} : !smt.bv<32>

  // CHECK: %{{.*}} = smt.not %{{.*}} {smt.some_attr}
  %6 = smt.not %a {smt.some_attr}
  // CHECK: %{{.*}} = smt.and %{{.*}}, %{{.*}}, %{{.*}} {smt.some_attr}
  %7 = smt.and %a, %a, %a {smt.some_attr}
  // CHECK: %{{.*}} = smt.or %{{.*}}, %{{.*}}, %{{.*}} {smt.some_attr}
  %8 = smt.or %a, %a, %a {smt.some_attr}
  // CHECK: %{{.*}} = smt.xor %{{.*}}, %{{.*}}, %{{.*}} {smt.some_attr}
  %9 = smt.xor %a, %a, %a {smt.some_attr}
  // CHECK: %{{.*}} = smt.implies %{{.*}}, %{{.*}} {smt.some_attr}
  %10 = smt.implies %a, %a {smt.some_attr}

  // CHECK: smt.apply_func %{{.*}}(%{{.*}}, %{{.*}}) {smt.some_attr} : !smt.func<(!smt.int, !smt.bool) !smt.bool>
  %11 = smt.apply_func %e(%c, %a) {smt.some_attr} : !smt.func<(!smt.int, !smt.bool) !smt.bool>

  return
}

// CHECK-LABEL: func @quantifiers
func.func @quantifiers() {
  // CHECK-NEXT: smt.forall ["a", "b"] weight 2 attributes {smt.some_attr} {
  // CHECK-NEXT: ^bb0({{.*}}: !smt.bool, {{.*}}: !smt.bool):
  // CHECK-NEXT:   smt.eq
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: } patterns {
  // CHECK-NEXT: ^bb0(%{{.*}}: !smt.bool, %{{.*}}: !smt.bool):
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: }, {
  // CHECK-NEXT: ^bb0(%{{.*}}: !smt.bool):
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: }
  %0 = smt.forall ["a", "b"] weight 2 attributes {smt.some_attr} {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    %1 = smt.eq %arg2, %arg3 : !smt.bool
    smt.yield %1 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2, %arg3 : !smt.bool, !smt.bool
  }, {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2, %arg3 : !smt.bool, !smt.bool
  }

  // CHECK-NEXT: smt.forall ["a", "b"] no_pattern attributes {smt.some_attr} {
  // CHECK-NEXT: ^bb0({{.*}}: !smt.bool, {{.*}}: !smt.bool):
  // CHECK-NEXT:   smt.eq
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: }
  %1 = smt.forall ["a", "b"] no_pattern attributes {smt.some_attr} {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    %2 = smt.eq %arg2, %arg3 : !smt.bool
    smt.yield %2 : !smt.bool
  }

  // CHECK-NEXT: smt.forall {
  // CHECK-NEXT:   smt.constant
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: }
  %2 = smt.forall {
    %3 = smt.constant true
    smt.yield %3 : !smt.bool
  }

  // CHECK-NEXT: smt.exists ["a", "b"] weight 2 attributes {smt.some_attr} {
  // CHECK-NEXT: ^bb0({{.*}}: !smt.bool, {{.*}}: !smt.bool):
  // CHECK-NEXT:   smt.eq
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: } patterns {
  // CHECK-NEXT: ^bb0(%{{.*}}: !smt.bool, %{{.*}}: !smt.bool):
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: }, {
  // CHECK-NEXT: ^bb0(%{{.*}}: !smt.bool):
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: }
  %3 = smt.exists ["a", "b"] weight 2 attributes {smt.some_attr} {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    %4 = smt.eq %arg2, %arg3 : !smt.bool
    smt.yield %4 : !smt.bool {smt.some_attr}
  } patterns {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2, %arg3 : !smt.bool, !smt.bool
  }, {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2, %arg3 : !smt.bool, !smt.bool
  }

  // CHECK-NEXT: smt.exists no_pattern attributes {smt.some_attr} {
  // CHECK-NEXT: ^bb0({{.*}}: !smt.bool, {{.*}}: !smt.bool):
  // CHECK-NEXT:   smt.eq
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: }
  %4 = smt.exists no_pattern attributes {smt.some_attr} {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    %5 = smt.eq %arg2, %arg3 : !smt.bool
    smt.yield %5 : !smt.bool {smt.some_attr}
  }

  // CHECK-NEXT: smt.exists [] {
  // CHECK-NEXT:   smt.constant
  // CHECK-NEXT:   smt.yield %{{.*}}
  // CHECK-NEXT: }
  %5 = smt.exists [] {
    %6 = smt.constant true
    smt.yield %6 : !smt.bool
  }

  return
}
