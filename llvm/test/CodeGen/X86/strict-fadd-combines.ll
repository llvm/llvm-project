; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

define float @fneg_strict_fadd_to_strict_fsub(float %x, float %y) {
  ; CHECK: subss %{{.*}}, %{{.*}}
  ; CHECK-NEXT: retq
  %neg = fneg float %y
  %add = call float @llvm.experimental.constrained.fadd.f32(float %x, float %neg, metadata!"round.dynamic", metadata!"fpexcept.strict")
  ret float %add
}

define float @fneg_strict_fadd_to_strict_fsub_2(float %x, float %y) {
  ; CHECK: subss %{{.*}}, %{{.*}}
  ; CHECK-NEXT: retq
  %neg = fneg float %y
  %add = call float @llvm.experimental.constrained.fadd.f32(float %neg, float %x, metadata!"round.dynamic", metadata!"fpexcept.strict")
  ret float %add
}

define double @fneg_strict_fadd_to_strict_fsub_d(double %x, double %y) {
  ; CHECK: subsd %{{.*}}, %{{.*}}
  ; CHECK-NEXT: retq
  %neg = fneg double %y
  %add = call double @llvm.experimental.constrained.fadd.f64(double %x, double %neg, metadata!"round.dynamic", metadata!"fpexcept.strict")
  ret double %add
}

define double @fneg_strict_fadd_to_strict_fsub_2d(double %x, double %y) {
  ; CHECK: subsd %{{.*}}, %{{.*}}
  ; CHECK-NEXT: retq
  %neg = fneg double %y
  %add = call double @llvm.experimental.constrained.fadd.f64(double %neg, double %x, metadata!"round.dynamic", metadata!"fpexcept.strict")
  ret double %add
}


declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
