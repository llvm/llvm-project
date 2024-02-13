// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(test-transform-dialect-interpreter{transform-file-name=%p%{fs-sep}include%{fs-sep}test-interpreter-external-concurrent-source.mlir}))" \
// RUN:             --verify-diagnostics

// Exercising the pass on multiple functions of different lengths that may be
// processed concurrently. This should expose potential races.

func.func @f1() {
  // expected-remark @below {{matched}}
  return
}

func.func @f2() {
  // expected-remark @below {{matched}}
  return
}

func.func @f3() {
  call @f2() : () -> ()
  call @f2() : () -> ()
  call @f5() : () -> ()
  call @f7() : () -> ()
  call @f5() : () -> ()
  call @f5() : () -> ()
  // expected-remark @below {{matched}}
  return
}

func.func @f4() {
  call @f3() : () -> ()
  call @f3() : () -> ()
  // expected-remark @below {{matched}}
  return
}

func.func @f5() {
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f1() : () -> ()
  call @f1() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  // expected-remark @below {{matched}}
  return
}

func.func @f6() {
  // expected-remark @below {{matched}}
  return
}

func.func @f7() {
  // expected-remark @below {{matched}}
  return
}
