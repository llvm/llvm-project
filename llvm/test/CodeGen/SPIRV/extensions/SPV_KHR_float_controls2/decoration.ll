; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: Capability FloatControls2
; CHECK: Extension "SPV_KHR_float_controls2"

; CHECK: OpName %[[#addRes:]] "addRes"
; CHECK: OpName %[[#subRes:]] "subRes"
; CHECK: OpName %[[#mulRes:]] "mulRes"
; CHECK: OpName %[[#divRes:]] "divRes"
; CHECK: OpName %[[#remRes:]] "remRes"
; CHECK: OpName %[[#negRes:]] "negRes"
; CHECK: OpName %[[#oeqRes:]] "oeqRes"
; CHECK: OpName %[[#oneRes:]] "oneRes"
; CHECK: OpName %[[#oltRes:]] "oltRes"
; CHECK: OpName %[[#ogtRes:]] "ogtRes"
; CHECK: OpName %[[#oleRes:]] "oleRes"
; CHECK: OpName %[[#ogeRes:]] "ogeRes"
; CHECK: OpName %[[#ordRes:]] "ordRes"
; CHECK: OpName %[[#ueqRes:]] "ueqRes"
; CHECK: OpName %[[#uneRes:]] "uneRes"
; CHECK: OpName %[[#ultRes:]] "ultRes"
; CHECK: OpName %[[#ugtRes:]] "ugtRes"
; CHECK: OpName %[[#uleRes:]] "uleRes"
; CHECK: OpName %[[#ugeRes:]] "ugeRes"
; CHECK: OpName %[[#unoRes:]] "unoRes"
; CHECK: OpName %[[#modRes:]] "modRes"
; CHECK: OpName %[[#maxRes:]] "maxRes"
; CHECK: OpName %[[#maxCommonRes:]] "maxCommonRes"
; CHECK: OpName %[[#addResV:]] "addResV"
; CHECK: OpName %[[#subResV:]] "subResV"
; CHECK: OpName %[[#mulResV:]] "mulResV"
; CHECK: OpName %[[#divResV:]] "divResV"
; CHECK: OpName %[[#remResV:]] "remResV"
; CHECK: OpName %[[#negResV:]] "negResV"
; CHECK: OpName %[[#oeqResV:]] "oeqResV"
; CHECK: OpName %[[#oneResV:]] "oneResV"
; CHECK: OpName %[[#oltResV:]] "oltResV"
; CHECK: OpName %[[#ogtResV:]] "ogtResV"
; CHECK: OpName %[[#oleResV:]] "oleResV"
; CHECK: OpName %[[#ogeResV:]] "ogeResV"
; CHECK: OpName %[[#ordResV:]] "ordResV"
; CHECK: OpName %[[#ueqResV:]] "ueqResV"
; CHECK: OpName %[[#uneResV:]] "uneResV"
; CHECK: OpName %[[#ultResV:]] "ultResV"
; CHECK: OpName %[[#ugtResV:]] "ugtResV"
; CHECK: OpName %[[#uleResV:]] "uleResV"
; CHECK: OpName %[[#ugeResV:]] "ugeResV"
; CHECK: OpName %[[#unoResV:]] "unoResV"
; CHECK: OpName %[[#modResV:]] "modResV"
; CHECK: OpName %[[#maxResV:]] "maxResV"
; CHECK: OpName %[[#maxCommonResV:]] "maxCommonResV"
; CHECK: OpDecorate %[[#subRes]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#mulRes]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#divRes]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#remRes]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#negRes]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#oeqRes]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#oltRes]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#ogtRes]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#oleRes]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#ogeRes]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#ordRes]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#ueqRes]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#maxRes]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#maxCommonRes]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#subResV]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#mulResV]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#divResV]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#remResV]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#negResV]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#oeqResV]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#oltResV]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#ogtResV]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#oleResV]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#ogeResV]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#ordResV]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#ueqResV]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#maxResV]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#maxCommonResV]] FPFastMathMode NotNaN|NotInf

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func float @_Z4fmodff(float, float)
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z23__spirv_ocl_fmax_commonff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1
declare spir_func <2 x float> @_Z4fmodDv2_fDv2_f(<2 x float>, <2 x float>)
declare dso_local spir_func noundef nofpclass(nan inf) <2 x float> @_Z16__spirv_ocl_fmaxDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf), <2 x float> noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) <2 x float> @_Z23__spirv_ocl_fmax_commonDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf), <2 x float> noundef nofpclass(nan inf)) local_unnamed_addr #1

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @foo(float %1, float %2) {
entry:
  %addRes = fadd float %1,  %2
  %subRes = fsub nnan float %1,  %2
  %mulRes = fmul ninf float %1,  %2
  %divRes = fdiv nsz float %1,  %2
  %remRes = frem arcp float %1,  %2
  %negRes = fneg fast float %1
  %oeqRes = fcmp nnan ninf oeq float %1,  %2
  %oneRes = fcmp one float %1,  %2, !spirv.Decorations !3
  %oltRes = fcmp nnan olt float %1,  %2, !spirv.Decorations !3
  %ogtRes = fcmp ninf ogt float %1,  %2, !spirv.Decorations !3
  %oleRes = fcmp nsz ole float %1,  %2, !spirv.Decorations !3
  %ogeRes = fcmp arcp oge float %1,  %2, !spirv.Decorations !3
  %ordRes = fcmp fast ord float %1,  %2, !spirv.Decorations !3
  %ueqRes = fcmp nnan ninf ueq float %1,  %2, !spirv.Decorations !3
  %uneRes = fcmp une float %1,  %2, !spirv.Decorations !3
  %ultRes = fcmp ult float %1,  %2, !spirv.Decorations !3
  %ugtRes = fcmp ugt float %1,  %2, !spirv.Decorations !3
  %uleRes = fcmp ule float %1,  %2, !spirv.Decorations !3
  %ugeRes = fcmp uge float %1,  %2, !spirv.Decorations !3
  %unoRes = fcmp uno float %1,  %2, !spirv.Decorations !3
  %modRes = call spir_func float @_Z4fmodff(float %1, float %2)
  %maxRes = tail call fast spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
   %maxCommonRes = tail call spir_func noundef float @_Z23__spirv_ocl_fmax_commonff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  ret void
}

define weak_odr dso_local spir_kernel void @fooV(<2 x float> %v1, <2 x float> %v2) {
  %addResV = fadd <2 x float> %v1,  %v2
  %subResV = fsub nnan <2 x float> %v1,  %v2
  %mulResV = fmul ninf <2 x float> %v1,  %v2
  %divResV = fdiv nsz <2 x float> %v1,  %v2
  %remResV = frem arcp <2 x float> %v1,  %v2
  %negResV = fneg fast <2 x float> %v1
  %oeqResV = fcmp nnan ninf oeq <2 x float> %v1,  %v2
  %oneResV = fcmp one <2 x float> %v1,  %v2, !spirv.Decorations !3
  %oltResV = fcmp nnan olt <2 x float> %v1,  %v2, !spirv.Decorations !3
  %ogtResV = fcmp ninf ogt <2 x float> %v1,  %v2, !spirv.Decorations !3
  %oleResV = fcmp nsz ole <2 x float> %v1,  %v2, !spirv.Decorations !3
  %ogeResV = fcmp arcp oge <2 x float> %v1,  %v2, !spirv.Decorations !3
  %ordResV = fcmp fast ord <2 x float> %v1,  %v2, !spirv.Decorations !3
  %ueqResV = fcmp nnan ninf ueq <2 x float> %v1,  %v2, !spirv.Decorations !3
  %uneResV = fcmp une <2 x float> %v1,  %v2, !spirv.Decorations !3
  %ultResV = fcmp ult <2 x float> %v1,  %v2, !spirv.Decorations !3
  %ugtResV = fcmp ugt <2 x float> %v1,  %v2, !spirv.Decorations !3
  %uleResV = fcmp ule <2 x float> %v1,  %v2, !spirv.Decorations !3
  %ugeResV = fcmp uge <2 x float> %v1,  %v2, !spirv.Decorations !3
  %unoResV = fcmp uno <2 x float> %v1,  %v2, !spirv.Decorations !3
  %modResV = call spir_func <2 x float> @_Z4fmodDv2_fDv2_f(<2 x float> %v1, <2 x float> %v2)
  %maxResV = tail call fast spir_func noundef nofpclass(nan inf) <2 x float> @_Z16__spirv_ocl_fmaxDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
   %maxCommonResV = tail call spir_func noundef <2 x float> @_Z23__spirv_ocl_fmax_commonDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
  ret void
}

!3 = !{!5, !4}
!4 = !{i32 42} ; 42 is NoContraction decoration
!5 = !{i32 40, i32 393216} ; 40 is FPFastMathMode
