// RUN: mlir-translate --export-smtlib %s | FileCheck %s

smt.solver () : () -> () {
  %0 = smt.int.constant 5
  %1 = smt.int.constant 10

  // CHECK: (assert (let (([[V0:.+]] (+ 5 5 5)))
  // CHECK:         (let (([[V1:.+]] (= [[V0]] 10)))
  // CHECK:         [[V1]])))

  // CHECK-INLINED: (assert (= (+ 5 5 5) 10))
  %2 = smt.int.add %0, %0, %0
  %a2 = smt.eq %2, %1 : !smt.int
  smt.assert %a2

  // CHECK: (assert (let (([[V2:.+]] (* 5 5 5)))
  // CHECK:         (let (([[V3:.+]] (= [[V2]] 10)))
  // CHECK:         [[V3]])))

  // CHECK-INLINED: (assert (= (* 5 5 5) 10))
  %3 = smt.int.mul %0, %0, %0
  %a3 = smt.eq %3, %1 : !smt.int
  smt.assert %a3

  // CHECK: (assert (let (([[V4:.+]] (- 5 5)))
  // CHECK:         (let (([[V5:.+]] (= [[V4]] 10)))
  // CHECK:         [[V5]])))

  // CHECK-INLINED: (assert (= (- 5 5) 10))
  %4 = smt.int.sub %0, %0
  %a4 = smt.eq %4, %1 : !smt.int
  smt.assert %a4

  // CHECK: (assert (let (([[V6:.+]] (div 5 5)))
  // CHECK:         (let (([[V7:.+]] (= [[V6]] 10)))
  // CHECK:         [[V7]])))

  // CHECK-INLINED: (assert (= (div 5 5) 10))
  %5 = smt.int.div %0, %0
  %a5 = smt.eq %5, %1 : !smt.int
  smt.assert %a5

  // CHECK: (assert (let (([[V8:.+]] (mod 5 5)))
  // CHECK:         (let (([[V9:.+]] (= [[V8]] 10)))
  // CHECK:         [[V9]])))

  // CHECK-INLINED: (assert (= (mod 5 5) 10))
  %6 = smt.int.mod %0, %0
  %a6 = smt.eq %6, %1 : !smt.int
  smt.assert %a6

  // CHECK: (assert (let (([[V10:.+]] (<= 5 5)))
  // CHECK:         [[V10]]))

  // CHECK-INLINED: (assert (<= 5 5))
  %9 = smt.int.cmp le %0, %0
  smt.assert %9

  // CHECK: (assert (let (([[V11:.+]] (< 5 5)))
  // CHECK:         [[V11]]))

  // CHECK-INLINED: (assert (< 5 5))
  %10 = smt.int.cmp lt %0, %0
  smt.assert %10

  // CHECK: (assert (let (([[V12:.+]] (>= 5 5)))
  // CHECK:         [[V12]]))

  // CHECK-INLINED: (assert (>= 5 5))
  %11 = smt.int.cmp ge %0, %0
  smt.assert %11

  // CHECK: (assert (let (([[V13:.+]] (> 5 5)))
  // CHECK:         [[V13]]))

  // CHECK-INLINED: (assert (> 5 5))
  %12 = smt.int.cmp gt %0, %0
  smt.assert %12

  // CHECK: (reset)
  // CHECK-INLINED: (reset)
}
