// RUN: mlir-translate --export-smtlib %s | FileCheck %s
// RUN: mlir-translate --export-smtlib --smtlibexport-inline-single-use-values %s | FileCheck %s --check-prefix=CHECK-INLINED

smt.solver () : () -> () {
  %c = smt.int.constant 0
  %true = smt.constant true

  // CHECK: (assert (let (([[V0:.+]] ((as const (Array Int Bool)) true)))
  // CHECK:         (let (([[V1:.+]] (store [[V0]] 0 true)))
  // CHECK:         (let (([[V2:.+]] (select [[V1]] 0)))
  // CHECK:         [[V2]]))))

  // CHECK-INLINED: (assert (select (store ((as const (Array Int Bool)) true) 0 true) 0))
  %0 = smt.array.broadcast %true : !smt.array<[!smt.int -> !smt.bool]>
  %1 = smt.array.store %0[%c], %true : !smt.array<[!smt.int -> !smt.bool]>
  %2 = smt.array.select %1[%c] : !smt.array<[!smt.int -> !smt.bool]>
  smt.assert %2

  // CHECK: (reset)
  // CHECK-INLINED: (reset)
}
