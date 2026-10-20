; RUN: llc < %s -mtriple=hexagon-unknown-linux-musl \
; RUN:      | FileCheck %s -check-prefix=CHECK

define i64 @double_to_i128(double %d) nounwind strictfp {
; CHECK-LABEL: double_to_i128:
; CHECK:       // %bb.0:
; CHECK:          call __fixdfti
; CHECK:          dealloc_return
  %1 = tail call i128 @llvm.experimental.constrained.fptosi.i128.f64(double %d, metadata !"fpexcept.strict")
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @double_to_ui128(double %d) nounwind strictfp {
; CHECK-LABEL: double_to_ui128:
; CHECK:       // %bb.0:
; CHECK:          call __fixunsdfti
; CHECK:          dealloc_return
  %1 = tail call i128 @llvm.experimental.constrained.fptoui.i128.f64(double %d, metadata !"fpexcept.strict")
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @float_to_i128(float %d) nounwind strictfp {
; CHECK-LABEL: float_to_i128:
; CHECK:       // %bb.0:
; CHECK:          call __fixsfti
; CHECK:          dealloc_return
  %1 = tail call i128 @llvm.experimental.constrained.fptosi.i128.f32(float %d, metadata !"fpexcept.strict")
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @float_to_ui128(float %d) nounwind strictfp {
; CHECK-LABEL: float_to_ui128:
; CHECK:       // %bb.0:
; CHECK:         call __fixunssfti
; CHECK:         dealloc_return
  %1 = tail call i128 @llvm.experimental.constrained.fptoui.i128.f32(float %d, metadata !"fpexcept.strict")
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define double @ui128_to_double(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: ui128_to_double:
; CHECK:       // %bb.0:
; CHECK:         call __floatuntidf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call double @llvm.experimental.constrained.uitofp.f64.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret double %3
}

define float @i128_to_float(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: i128_to_float:
; CHECK:       // %bb.0:
; CHECK:         call __floattisf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call float @llvm.experimental.constrained.sitofp.f32.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %3
}

define float @ui128_to_float(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: ui128_to_float:
; CHECK:       // %bb.0:
; CHECK:         call __floatuntisf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call float @llvm.experimental.constrained.uitofp.f32.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %3
}

declare i128 @llvm.experimental.constrained.fptosi.i128.f64(double, metadata)
declare i128 @llvm.experimental.constrained.fptoui.i128.f64(double, metadata)
declare i128 @llvm.experimental.constrained.fptosi.i128.f32(float, metadata)
declare i128 @llvm.experimental.constrained.fptoui.i128.f32(float, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i128(i128, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i128(i128, metadata, metadata)
declare float @llvm.experimental.constrained.sitofp.f32.i128(i128, metadata, metadata)
declare float @llvm.experimental.constrained.uitofp.f32.i128(i128, metadata, metadata)
