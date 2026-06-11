// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @binops
llvm.func @binops(%a: f32, %b: f32) {
  // CHECK: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.ignore")
  %0 = llvm.fadd %a, %b {fenv = #llvm.fenv<rounding_mode = upward>} : f32
  // CHECK: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.downward", metadata !"fpexcept.ignore")
  %1 = llvm.fsub %a, %b {fenv = #llvm.fenv<rounding_mode = downward>} : f32
  // CHECK: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  %2 = llvm.fmul %a, %b {fenv = #llvm.fenv<rounding_mode = tonearest>} : f32
  // CHECK: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
  %3 = llvm.fdiv %a, %b {fenv = #llvm.fenv<rounding_mode = upwardzero>} : f32
  // CHECK: call float @llvm.experimental.constrained.frem.f32(float %{{.*}}, float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.ignore")
  %4 = llvm.frem %a, %b {fenv = #llvm.fenv<rounding_mode = upward>} : f32
  llvm.return
}

// -----

// `fcmp` lowering chooses between the signaling and quiet constrained
// comparisons (or a plain `fcmp`) based on the exception mode, the
// `strict_snan` flag, and the predicate.
// CHECK-LABEL: define void @cmp
llvm.func @cmp(%a: f32, %b: f32) {
  // Unmasked exceptions default `strict_snan` to true, so a relational predicate
  // becomes a signaling comparison.
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  %0 = llvm.fcmp "olt" %a, %b {fenv = #llvm.fenv<except_mode = unmasked>} : f32

  // Equality predicates are non-signaling even with strict sNaN handling.
  // CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  %1 = llvm.fcmp "oeq" %a, %b {fenv = #llvm.fenv<except_mode = unmasked>} : f32

  // With `strict_snan = false`, even a relational predicate stays quiet.
  // CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt", metadata !"fpexcept.maytrap")
  %2 = llvm.fcmp "olt" %a, %b {fenv = #llvm.fenv<except_mode = unmasked, strict_snan = false, strict_except = false>} : f32
  llvm.return
}

// -----

// Masked exceptions emit a plain `fcmp`, even when another field (here a
// rounding mode, which is irrelevant to comparisons) makes the attribute
// non-default.
// CHECK-LABEL: define i1 @cmp_masked
llvm.func @cmp_masked(%a: f32, %b: f32) -> i1 {
  // CHECK: fcmp olt float %{{.*}}, %{{.*}}
  // CHECK-NOT: experimental.constrained
  %0 = llvm.fcmp "olt" %a, %b {fenv = #llvm.fenv<rounding_mode = upward>} : f32
  llvm.return %0 : i1
}

// -----

// `strict_except` controls whether unmasked/unknown exceptions map to
// `fpexcept.strict` (true, the default) or `fpexcept.maytrap` (false).
// CHECK-LABEL: define void @except_modes
llvm.func @except_modes(%a: f32, %b: f32) {
  // Explicit `strict_except = false` with an unmasked mode maps to maytrap.
  // CHECK: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  %0 = llvm.fadd %a, %b {fenv = #llvm.fenv<except_mode = unmasked, strict_except = false>} : f32
  // `unknown` is treated like `unmasked`, so it defaults to strict.
  // CHECK: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  %1 = llvm.fsub %a, %b {fenv = #llvm.fenv<except_mode = unknown>} : f32
  llvm.return
}

// -----

// CHECK-LABEL: define void @casts
llvm.func @casts(%a: f32, %b: f64) {
  // CHECK: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.ignore")
  %0 = llvm.fpext %a {fenv = #llvm.fenv<rounding_mode = upward>} : f32 to f64
  // CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(double %{{.*}}, metadata !"round.downward", metadata !"fpexcept.ignore")
  %1 = llvm.fptrunc %b {fenv = #llvm.fenv<rounding_mode = downward>} : f64 to f32
  llvm.return
}

// -----

// CHECK-LABEL: define float @math_intrinsic
llvm.func @math_intrinsic(%a: f32) -> f32 {
  // CHECK: call float @llvm.experimental.constrained.sqrt.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.ignore")
  %0 = llvm.intr.sqrt(%a) {fenv = #llvm.fenv<rounding_mode = upward>} : (f32) -> f32
  llvm.return %0 : f32
}

// -----

// A named intrinsic call is mapped to its constrained counterpart.
// CHECK-LABEL: define float @call_intrinsic
llvm.func @call_intrinsic(%a: f32, %b: f32) -> f32 {
  // CHECK: call float @llvm.experimental.constrained.maxnum.f32(float %{{.*}}, float %{{.*}}, metadata !"fpexcept.ignore")
  %0 = llvm.call_intrinsic "llvm.maxnum.f32"(%a, %b) {fenv = #llvm.fenv<rounding_mode = upward>} : (f32, f32) -> f32
  llvm.return %0 : f32
}

// -----

// An absent attribute lowers to the regular, unconstrained operation.
// CHECK-LABEL: define float @no_fenv
llvm.func @no_fenv(%a: f32, %b: f32) -> f32 {
  // CHECK: fadd float %{{.*}}, %{{.*}}
  // CHECK-NOT: experimental.constrained
  %0 = llvm.fadd %a, %b : f32
  llvm.return %0 : f32
}

// -----

// An empty attribute describes the default environment, but carrying the
// attribute still selects the constrained intrinsic (with the default rounding
// mode and exception behavior).
// CHECK-LABEL: define float @empty_fenv
llvm.func @empty_fenv(%a: f32, %b: f32) -> f32 {
  // CHECK: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  %0 = llvm.fadd %a, %b {fenv = #llvm.fenv<>} : f32
  llvm.return %0 : f32
}

// -----

// An attribute whose fields are all set to their defaults is normalized to the
// canonical (empty) environment, which is still present and therefore selects
// the constrained intrinsic with the default settings.
// CHECK-LABEL: define float @all_default_fenv
llvm.func @all_default_fenv(%a: f32, %b: f32) -> f32 {
  // CHECK: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  %0 = llvm.fadd %a, %b {fenv = #llvm.fenv<rounding_mode = dynamic,
                                           dynamic_rounding_mode = unknown,
                                           except_mode = masked,
                                           strict_snan = false,
                                           strict_except = false>} : f32
  llvm.return %0 : f32
}
