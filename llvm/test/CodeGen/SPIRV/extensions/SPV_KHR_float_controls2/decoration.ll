; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

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
; CHECK: OpDecorate %[[#subRes]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#mulRes]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#divRes]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#remRes]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#negRes]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#oeqRes]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#oneRes]] FPFastMathMode AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#oltRes]] FPFastMathMode NotNaN|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#ogtRes]] FPFastMathMode NotInf|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#oleRes]] FPFastMathMode NSZ|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#ogeRes]] FPFastMathMode AllowRecip|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#ordRes]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform 
; CHECK: OpDecorate %[[#ueqRes]] FPFastMathMode NotNaN|NotInf|AllowReassoc|AllowTransform 
; CHECK: OpDecorate %[[#uneRes]] FPFastMathMode AllowReassoc|AllowTransform 
; CHECK: OpDecorate %[[#ultRes]] FPFastMathMode AllowReassoc|AllowTransform 
; CHECK: OpDecorate %[[#ugtRes]] FPFastMathMode AllowReassoc|AllowTransform 
; CHECK: OpDecorate %[[#uleRes]] FPFastMathMode AllowReassoc|AllowTransform 
; CHECK: OpDecorate %[[#ugeRes]] FPFastMathMode AllowReassoc|AllowTransform 
; CHECK: OpDecorate %[[#unoRes]] FPFastMathMode AllowReassoc|AllowTransform 
; CHECK-NOT: OpDecorate %[[#modRes]] FPFastMathMode
; CHECK: OpDecorate %[[#maxRes]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform 
; CHECK: OpDecorate %[[#maxCommonRes:]] FPFastMathMode NotNaN|NotInf 

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func float @_Z4fmodff(float, float)
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z23__spirv_ocl_fmax_commonff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1

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

!3 = !{!5, !4}
!4 = !{i32 42} ; 42 is NoContraction decoration
!5 = !{i32 40, i32 393216} ; 40 is FPFastMathMode
