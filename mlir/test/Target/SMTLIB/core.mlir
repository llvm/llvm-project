// RUN: mlir-translate --export-smtlib %s | FileCheck %s
// RUN: mlir-translate --export-smtlib --smtlibexport-inline-single-use-values %s | FileCheck %s --check-prefix=CHECK-INLINED

smt.solver () : () -> () {
  %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  %true = smt.constant true
  %false = smt.constant false

  // CHECK: (declare-const b (_ BitVec 32))
  // CHECK: (assert (let (([[V0:.+]] (= #x00000000 b)))
  // CHECK:         [[V0]]))

  // CHECK-INLINED: (declare-const b (_ BitVec 32))
  // CHECK-INLINED: (assert (= #x00000000 b))
  %21 = smt.declare_fun "b" : !smt.bv<32>
  %23 = smt.eq %c0_bv32, %21 : !smt.bv<32>
  smt.assert %23

  // CHECK: (assert (let (([[V1:.+]] (distinct #x00000000 #x00000000)))
  // CHECK:         [[V1]]))

  // CHECK-INLINED: (assert (distinct #x00000000 #x00000000))
  %24 = smt.distinct %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %24

  // CHECK: (declare-const a Bool)
  // CHECK: (assert (let (([[V2:.+]] (ite a #x00000000 b)))
  // CHECK:         (let (([[V3:.+]] (= #x00000000 [[V2]])))
  // CHECK:         [[V3]])))

  // CHECK-INLINED: (declare-const a Bool)
  // CHECK-INLINED: (assert (= #x00000000 (ite a #x00000000 b)))
  %20 = smt.declare_fun "a" : !smt.bool
  %38 = smt.ite %20, %c0_bv32, %21 : !smt.bv<32>
  %4 = smt.eq %c0_bv32, %38 : !smt.bv<32>
  smt.assert %4

  // CHECK: (assert (let (([[V4:.+]] (not true)))
  // CHECK:         (let (([[V5:.+]] (and [[V4]] true false)))
  // CHECK:         (let (([[V6:.+]] (or [[V5]] [[V4]] true)))
  // CHECK:         (let (([[V7:.+]] (xor [[V4]] [[V6]])))
  // CHECK:         (let (([[V8:.+]] (=> [[V7]] false)))
  // CHECK:         [[V8]]))))))

  // CHECK-INLINED: (assert (let (([[V15:.+]] (not true)))
  // CHECK-INLINED:         (=> (xor [[V15]] (or (and [[V15]] true false) [[V15]] true)) false)))
  %39 = smt.not %true
  %40 = smt.and %39, %true, %false
  %41 = smt.or %40, %39, %true
  %42 = smt.xor %39, %41
  %43 = smt.implies %42, %false
  smt.assert %43

  // CHECK: (declare-fun func1 (Bool Bool) Bool)
  // CHECK: (assert (let (([[V9:.+]] (func1 true false)))
  // CHECK:         [[V9]]))

  // CHECK-INLINED: (declare-fun func1 (Bool Bool) Bool)
  // CHECK-INLINED: (assert (func1 true false))
  %44 = smt.declare_fun "func1" : !smt.func<(!smt.bool, !smt.bool) !smt.bool>
  %45 = smt.apply_func %44(%true, %false) : !smt.func<(!smt.bool, !smt.bool) !smt.bool>
  smt.assert %45

  // CHECK: (assert (let (([[V10:.+]] (forall (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK:                           (let (([[V11:.+]] (= [[A]] [[B]])))
  // CHECK:                           [[V11]]))))
  // CHECK:         [[V10]]))

  // CHECK-INLINED: (assert (forall (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK-INLINED:                 (= [[A]] [[B]])))
  %1 = smt.forall ["a", "b"] {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %2 : !smt.bool
  }
  smt.assert %1

  // CHECK: (assert (let (([[V12:.+]] (exists (([[V13:.+]] Int) ([[V14:.+]] Int))
  // CHECK:                           (let (([[V15:.+]] (= [[V13]] [[V14]])))
  // CHECK:                           [[V15]]))))
  // CHECK:         [[V12]]))

  // CHECK-INLINED: (assert (exists (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK-INLINED:                 (= [[A]] [[B]])))
  %2 = smt.exists {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %3 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %3 : !smt.bool
  }
  smt.assert %2

  // Test: make sure that open parens from outside quantifier bodies are not
  // propagated into the body.
  // CHECK: (assert (let (([[V15:.+]] (exists (([[V16:.+]] Int) ([[V17:.+]] Int)){{$}}
  // CHECK:                                   (let (([[V18:.+]] (= [[V16]] [[V17]]))){{$}}
  // CHECK:                                   [[V18]])))){{$}}
  // CHECK:         (let (([[V19:.+]] (exists (([[V20:.+]] Int) ([[V21:.+]] Int)){{$}}
  // CHECK:                                   (let (([[V22:.+]] (= [[V20]] [[V21]]))){{$}}
  // CHECK:                                   [[V22]])))){{$}}
  // CHECK:         (let (([[V23:.+]] (and [[V19]] [[V15]]))){{$}}
  // CHECK:         [[V23]])))){{$}}
  %3 = smt.exists {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %5 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %5 : !smt.bool
  }
  %5 = smt.exists {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %6 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %6 : !smt.bool
  }
  %6 = smt.and %3, %5
  smt.assert %6

  // CHECK: (check-sat)
  // CHECK-INLINED: (check-sat)
  smt.check sat {} unknown {} unsat {}

  // CHECK: (reset)
  // CHECK-INLINED: (reset)
  smt.reset

  // CHECK: (push 1)
  // CHECK-INLINED: (push 1)
  smt.push 1

  // CHECK: (pop 1)
  // CHECK-INLINED: (pop 1)
  smt.pop 1

  // CHECK: (set-logic AUFLIA)
  // CHECK-INLINED: (set-logic AUFLIA)
  smt.set_logic "AUFLIA"

  // CHECK: (reset)
  // CHECK-INLINED: (reset)
}
