// RUN: mlir-opt %s --cse | FileCheck %s

func.func @declare_const_cse(%in: i8) -> (!smt.bool, !smt.bool){
  // CHECK: smt.declare_fun "a" : !smt.bool
  %a = smt.declare_fun "a" : !smt.bool
  // CHECK-NEXT: smt.declare_fun "a" : !smt.bool
  %b = smt.declare_fun "a" : !smt.bool
  // CHECK-NEXT: return
  %c = smt.declare_fun "a" : !smt.bool

  return %a, %b : !smt.bool, !smt.bool
}
