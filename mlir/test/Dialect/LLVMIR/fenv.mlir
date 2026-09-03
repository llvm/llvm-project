// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @roundtrip
llvm.func @roundtrip(%a: f32, %b: f32) {
  // CHECK: llvm.fadd %{{.*}}, %{{.*}} {fenv = #llvm.fenv<rounding_mode = upward>} : f32
  %0 = llvm.fadd %a, %b {fenv = #llvm.fenv<rounding_mode = upward>} : f32
  // `strict_snan`, like `strict_except`, is only meaningful (and thus retained)
  // when exceptions are not masked, where it defaults to `true`; an explicit
  // `false` round-trips.
  // CHECK: llvm.fmul %{{.*}}, %{{.*}} {fenv = #llvm.fenv<except_mode = unmasked, strict_snan = false>} : f32
  %1 = llvm.fmul %a, %b {fenv = #llvm.fenv<except_mode = unmasked, strict_snan = false>} : f32
  // `strict_except` is only meaningful (and thus retained) when exceptions are
  // not masked, where it defaults to `true`; an explicit `false` round-trips.
  // CHECK: llvm.fcmp "olt" %{{.*}}, %{{.*}} {fenv = #llvm.fenv<dynamic_rounding_mode = tonearest, except_mode = unmasked, strict_except = false>} : f32
  %2 = llvm.fcmp "olt" %a, %b {fenv = #llvm.fenv<dynamic_rounding_mode = tonearest, except_mode = unmasked, strict_except = false>} : f32
  llvm.return
}

// -----

// An attribute whose fields are all at their defaults normalizes to the
// canonical, empty `#llvm.fenv<>`. The attribute is still present (and so is
// preserved when printing): carrying it is not the same as having no attribute.

// CHECK-LABEL: @normalized
llvm.func @normalized(%a: f32, %b: f32) {
  // CHECK: llvm.fadd %{{.*}}, %{{.*}} {fenv = #llvm.fenv<>} : f32
  %0 = llvm.fadd %a, %b {fenv = #llvm.fenv<>} : f32
  // CHECK: llvm.fsub %{{.*}}, %{{.*}} {fenv = #llvm.fenv<>} : f32
  %1 = llvm.fsub %a, %b {fenv = #llvm.fenv<rounding_mode = dynamic,
                                           dynamic_rounding_mode = unknown,
                                           except_mode = masked,
                                           strict_snan = false,
                                           strict_except = false>} : f32
  llvm.return
}
