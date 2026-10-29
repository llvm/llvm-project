; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32r6 < %s | FileCheck %s
; RUN: llc -mtriple=mips64el-linux-gnu -mcpu=mips64r6 < %s | FileCheck %s

; The results are unused: only strict FP exception semantics keep the rints
; alive and ordered around the independent volatile store.
define void @strict_order_f32(float %x, float %y, ptr %p) strictfp {
; CHECK-LABEL: strict_order_f32:
; CHECK-NOT:   sw $
; CHECK:       rint.s $f{{[0-9]+}}, $f12
; CHECK-NOT:   rint.s
; CHECK:       sw ${{[0-9]+}}, 0(${{[0-9]+}})
; CHECK-NOT:   sw $
; CHECK:       rint.s $f{{[0-9]+}}, $f1{{[34]}}
; CHECK-NOT:   rint.s
; CHECK:       .end strict_order_f32
  %a = call float @llvm.experimental.constrained.rint.f32(float %x, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  store volatile i32 42, ptr %p
  %b = call float @llvm.experimental.constrained.rint.f32(float %y, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

define void @strict_order_f64(double %x, double %y, ptr %p) strictfp {
; CHECK-LABEL: strict_order_f64:
; CHECK-NOT:   sw $
; CHECK:       rint.d $f{{[0-9]+}}, $f12
; CHECK-NOT:   rint.d
; CHECK:       sw ${{[0-9]+}}, 0(${{[0-9]+}})
; CHECK-NOT:   sw $
; CHECK:       rint.d $f{{[0-9]+}}, $f1{{[34]}}
; CHECK-NOT:   rint.d
; CHECK:       .end strict_order_f64
  %a = call double @llvm.experimental.constrained.rint.f64(double %x, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  store volatile i32 42, ptr %p
  %b = call double @llvm.experimental.constrained.rint.f64(double %y, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}
