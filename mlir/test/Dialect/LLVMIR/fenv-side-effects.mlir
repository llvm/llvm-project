// RUN: mlir-opt %s -canonicalize | FileCheck %s

// Operations carrying a `#llvm.fenv` attribute are no longer unconditionally
// `Pure`. They only have an observable side effect when an exception may be
// raised (the exception mode is not `masked`) and that exception must be
// preserved (`strict_except` is set). Otherwise the (dead) operation can be
// eliminated, even when it is not speculatable.

// CHECK-LABEL: @dce_dead_ops
llvm.func @dce_dead_ops(%a: f32, %b: f32) {
  // A default (absent) floating-point environment behaves like a pure op and is
  // removed when its result is unused.
  // CHECK-NOT: llvm.fadd
  %0 = llvm.fadd %a, %b : f32

  // Masked exceptions: no side effect, removed.
  // CHECK-NOT: llvm.fmul
  %1 = llvm.fmul %a, %b {fenv = #llvm.fenv<except_mode = masked>} : f32

  // Non-masked exceptions explicitly not required to be preserved
  // (`strict_except = false`): the effect is ignorable, so the dead op is
  // removed even though it is not speculatable.
  // CHECK-NOT: llvm.fsub
  %2 = llvm.fsub %a, %b {fenv = #llvm.fenv<except_mode = unmasked, strict_except = false>} : f32

  // CHECK-NOT: llvm.fdiv
  %3 = llvm.fdiv %a, %b {fenv = #llvm.fenv<except_mode = unknown, strict_except = false>} : f32

  llvm.return
}

// -----

// Operations whose exceptions must be preserved carry a write effect and are
// not eliminated. With a non-masked exception mode, `strict_except` defaults to
// `true`, so it need not be requested explicitly.

// CHECK-LABEL: @keep_strict_except
llvm.func @keep_strict_except(%a: f32, %b: f32) {
  // Unmasked exceptions default to strict, so the dead op is preserved.
  // CHECK: llvm.fadd
  %0 = llvm.fadd %a, %b {fenv = #llvm.fenv<except_mode = unmasked>} : f32

  // `unknown` is treated like `unmasked` and likewise defaults to strict.
  // CHECK: llvm.fmul
  %1 = llvm.fmul %a, %b {fenv = #llvm.fenv<except_mode = unknown>} : f32

  // An explicit `strict_except = true` matches the default.
  // CHECK: llvm.fdiv
  %2 = llvm.fdiv %a, %b {fenv = #llvm.fenv<except_mode = unmasked, strict_except = true>} : f32

  llvm.return
}
