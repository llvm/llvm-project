; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

;; This test checks that the OpenCL.std instructions fmin_common, fmax_common are replaced with fmin, fmax with NInf and NNaN instead.

; CHECK-DAG: Capability FloatControls2
; CHECK: Extension "SPV_KHR_float_controls2"

; CHECK: OpName %[[#maxRes:]] "maxRes"
; CHECK: OpName %[[#maxCommonRes:]] "maxCommonRes"
; CHECK: OpName %[[#minRes:]] "minRes"
; CHECK: OpName %[[#minCommonRes:]] "minCommonRes"
; CHECK: OpName %[[#maxResV:]] "maxResV"
; CHECK: OpName %[[#maxCommonResV:]] "maxCommonResV"
; CHECK: OpName %[[#minResV:]] "minResV"
; CHECK: OpName %[[#minCommonResV:]] "minCommonResV"
; CHECK: OpDecorate %[[#maxRes]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#maxCommonRes]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#minRes]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#minCommonRes]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#maxResV]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#maxCommonResV]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#minResV]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#minCommonResV]] FPFastMathMode NotNaN|NotInf
; CHECK: %[[#maxRes]] = OpExtInst {{.*}} fmax
; CHECK: %[[#maxCommonRes]] = OpExtInst {{.*}} fmax
; CHECK: %[[#minRes]] = OpExtInst {{.*}} fmin
; CHECK: %[[#minCommonRes]] = OpExtInst {{.*}} fmin
; CHECK: %[[#maxResV]] = OpExtInst {{.*}} fmax
; CHECK: %[[#maxCommonResV]] = OpExtInst {{.*}} fmax
; CHECK: %[[#minResV]] = OpExtInst {{.*}} fmin
; CHECK: %[[#minCommonResV]] = OpExtInst {{.*}} fmin

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func float @_Z4fmodff(float, float)
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z23__spirv_ocl_fmax_commonff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fminff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z23__spirv_ocl_fmin_commonff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) <2 x float> @_Z16__spirv_ocl_fmaxDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf), <2 x float> noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) <2 x float> @_Z23__spirv_ocl_fmax_commonDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf), <2 x float> noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) <2 x float> @_Z16__spirv_ocl_fminDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf), <2 x float> noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) <2 x float> @_Z23__spirv_ocl_fmin_commonDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf), <2 x float> noundef nofpclass(nan inf)) local_unnamed_addr #1

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @foo(float %1, float %2) {
entry:
  %maxRes = tail call fast spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
   %maxCommonRes = tail call spir_func noundef float @_Z23__spirv_ocl_fmax_commonff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  %minRes = tail call fast spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fminff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
   %minCommonRes = tail call spir_func noundef float @_Z23__spirv_ocl_fmin_commonff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  ret void
}

define weak_odr dso_local spir_kernel void @fooV(<2 x float> %v1, <2 x float> %v2) {
  %maxResV = tail call fast spir_func noundef nofpclass(nan inf) <2 x float> @_Z16__spirv_ocl_fmaxDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
   %maxCommonResV = tail call spir_func noundef <2 x float> @_Z23__spirv_ocl_fmax_commonDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
  %minResV = tail call fast spir_func noundef nofpclass(nan inf) <2 x float> @_Z16__spirv_ocl_fminDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
   %minCommonResV = tail call spir_func noundef <2 x float> @_Z23__spirv_ocl_fmin_commonDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
  ret void
}
