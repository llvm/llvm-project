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

@G_addRes = global float 0.0
@G_subRes = global float 0.0
@G_mulRes = global float 0.0
@G_divRes = global float 0.0
@G_remRes = global float 0.0
@G_negRes = global float 0.0
@G_oeqRes = global i1 0
@G_oneRes = global i1 0
@G_oltRes = global i1 0
@G_ogtRes = global i1 0
@G_oleRes = global i1 0
@G_ogeRes = global i1 0
@G_ordRes = global i1 0
@G_ueqRes = global i1 0
@G_uneRes = global i1 0
@G_ultRes = global i1 0
@G_ugtRes = global i1 0
@G_uleRes = global i1 0
@G_ugeRes = global i1 0
@G_unoRes = global i1 0
@G_modRes = global float 0.0
@G_maxRes = global float 0.0
@G_maxCommonRes = global float 0.0

@G_addResV = global <2 x float> zeroinitializer
@G_subResV = global <2 x float> zeroinitializer
@G_mulResV = global <2 x float> zeroinitializer
@G_divResV = global <2 x float> zeroinitializer
@G_remResV = global <2 x float> zeroinitializer
@G_negResV = global <2 x float> zeroinitializer
@G_oeqResV = global <2 x i1> zeroinitializer
@G_oneResV = global <2 x i1> zeroinitializer
@G_oltResV = global <2 x i1> zeroinitializer
@G_ogtResV = global <2 x i1> zeroinitializer
@G_oleResV = global <2 x i1> zeroinitializer
@G_ogeResV = global <2 x i1> zeroinitializer
@G_ordResV = global <2 x i1> zeroinitializer
@G_ueqResV = global <2 x i1> zeroinitializer
@G_uneResV = global <2 x i1> zeroinitializer
@G_ultResV = global <2 x i1> zeroinitializer
@G_ugtResV = global <2 x i1> zeroinitializer
@G_uleResV = global <2 x i1> zeroinitializer
@G_ugeResV = global <2 x i1> zeroinitializer
@G_unoResV = global <2 x i1> zeroinitializer
@G_modResV = global <2 x float> zeroinitializer
@G_maxResV = global <2 x float> zeroinitializer
@G_maxCommonResV = global <2 x float> zeroinitializer

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
  store float %addRes, float* @G_addRes
  %subRes = fsub nnan float %1,  %2
  store float %subRes, float* @G_subRes
  %mulRes = fmul ninf float %1,  %2
  store float %mulRes, float* @G_mulRes
  %divRes = fdiv nsz float %1,  %2
  store float %divRes, float* @G_divRes
  %remRes = frem arcp float %1,  %2
  store float %remRes, float* @G_remRes
  %negRes = fneg fast float %1
  store float %negRes, float* @G_negRes
  %oeqRes = fcmp nnan ninf oeq float %1,  %2
  store i1 %oeqRes, i1* @G_oeqRes
  %oneRes = fcmp one float %1,  %2, !spirv.Decorations !3
  store i1 %oneRes, i1* @G_oneRes
  %oltRes = fcmp nnan olt float %1,  %2, !spirv.Decorations !3
  store i1 %oltRes, i1* @G_oltRes
  %ogtRes = fcmp ninf ogt float %1,  %2, !spirv.Decorations !3
  store i1 %ogtRes, i1* @G_ogtRes
  %oleRes = fcmp nsz ole float %1,  %2, !spirv.Decorations !3
  store i1 %oleRes, i1* @G_oleRes
  %ogeRes = fcmp arcp oge float %1,  %2, !spirv.Decorations !3
  store i1 %ogeRes, i1* @G_ogeRes
  %ordRes = fcmp fast ord float %1,  %2, !spirv.Decorations !3
  store i1 %ordRes, i1* @G_ordRes
  %ueqRes = fcmp nnan ninf ueq float %1,  %2, !spirv.Decorations !3
  store i1 %ueqRes, i1* @G_ueqRes
  %uneRes = fcmp une float %1,  %2, !spirv.Decorations !3
  store i1 %uneRes, i1* @G_uneRes
  %ultRes = fcmp ult float %1,  %2, !spirv.Decorations !3
  store i1 %ultRes, i1* @G_ultRes
  %ugtRes = fcmp ugt float %1,  %2, !spirv.Decorations !3
  store i1 %ugtRes, i1* @G_ugtRes
  %uleRes = fcmp ule float %1,  %2, !spirv.Decorations !3
  store i1 %uleRes, i1* @G_uleRes
  %ugeRes = fcmp uge float %1,  %2, !spirv.Decorations !3
  store i1 %ugeRes, i1* @G_ugeRes
  %unoRes = fcmp uno float %1,  %2, !spirv.Decorations !3
  store i1 %unoRes, i1* @G_unoRes
  %modRes = call spir_func float @_Z4fmodff(float %1, float %2)
  store float %modRes, float* @G_modRes
  %maxRes = tail call fast spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  store float %maxRes, float* @G_maxRes
   %maxCommonRes = tail call spir_func noundef float @_Z23__spirv_ocl_fmax_commonff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  store float %maxCommonRes, float* @G_maxCommonRes
  ret void
}

define weak_odr dso_local spir_kernel void @fooV(<2 x float> %v1, <2 x float> %v2) {
  %addResV = fadd <2 x float> %v1,  %v2
  store <2 x float> %addResV, <2 x float>* @G_addResV
  %subResV = fsub nnan <2 x float> %v1,  %v2
  store <2 x float> %subResV, <2 x float>* @G_subResV
  %mulResV = fmul ninf <2 x float> %v1,  %v2
  store <2 x float> %mulResV, <2 x float>* @G_mulResV
  %divResV = fdiv nsz <2 x float> %v1,  %v2
  store <2 x float> %divResV, <2 x float>* @G_divResV
  %remResV = frem arcp <2 x float> %v1,  %v2
  store <2 x float> %remResV, <2 x float>* @G_remResV
  %negResV = fneg fast <2 x float> %v1
  store <2 x float> %negResV, <2 x float>* @G_negResV
  %oeqResV = fcmp nnan ninf oeq <2 x float> %v1,  %v2
  store <2 x i1> %oeqResV, <2 x i1>* @G_oeqResV
  %oneResV = fcmp one <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %oneResV, <2 x i1>* @G_oneResV
  %oltResV = fcmp nnan olt <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %oltResV, <2 x i1>* @G_oltResV
  %ogtResV = fcmp ninf ogt <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %ogtResV, <2 x i1>* @G_ogtResV
  %oleResV = fcmp nsz ole <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %oleResV, <2 x i1>* @G_oleResV
  %ogeResV = fcmp arcp oge <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %ogeResV, <2 x i1>* @G_ogeResV
  %ordResV = fcmp fast ord <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %ordResV, <2 x i1>* @G_ordResV
  %ueqResV = fcmp nnan ninf ueq <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %ueqResV, <2 x i1>* @G_ueqResV
  %uneResV = fcmp une <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %uneResV, <2 x i1>* @G_uneResV
  %ultResV = fcmp ult <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %ultResV, <2 x i1>* @G_ultResV
  %ugtResV = fcmp ugt <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %ugtResV, <2 x i1>* @G_ugtResV
  %uleResV = fcmp ule <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %uleResV, <2 x i1>* @G_uleResV
  %ugeResV = fcmp uge <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %ugeResV, <2 x i1>* @G_ugeResV
  %unoResV = fcmp uno <2 x float> %v1,  %v2, !spirv.Decorations !3
  store <2 x i1> %unoResV, <2 x i1>* @G_unoResV
  %modResV = call spir_func <2 x float> @_Z4fmodDv2_fDv2_f(<2 x float> %v1, <2 x float> %v2)
  store <2 x float> %modResV, <2 x float>* @G_modResV
  %maxResV = tail call fast spir_func noundef nofpclass(nan inf) <2 x float> @_Z16__spirv_ocl_fmaxDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
  store <2 x float> %maxResV, <2 x float>* @G_maxResV
   %maxCommonResV = tail call spir_func noundef <2 x float> @_Z23__spirv_ocl_fmax_commonDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
  store <2 x float> %maxCommonResV, <2 x float>* @G_maxCommonResV
  ret void
}

!3 = !{!5, !4}
!4 = !{i32 42} ; 42 is NoContraction decoration
!5 = !{i32 40, i32 393216} ; 40 is FPFastMathMode
