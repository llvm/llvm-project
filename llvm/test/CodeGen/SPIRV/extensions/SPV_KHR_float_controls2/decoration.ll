; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: Capability FloatControls2
; CHECK: Extension "SPV_KHR_float_controls2"

; CHECK: OpDecorate %[[#subRes:]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#mulRes:]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#divRes:]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#remRes:]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#negRes:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#oeqRes:]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#oltRes:]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#ogtRes:]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#oleRes:]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#ogeRes:]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#ordRes:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#ueqRes:]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#maxRes:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#maxCommonRes:]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#subResV:]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#mulResV:]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#divResV:]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#remResV:]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#negResV:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#oeqResV:]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#oltResV:]] FPFastMathMode NotNaN
; CHECK: OpDecorate %[[#ogtResV:]] FPFastMathMode NotInf
; CHECK: OpDecorate %[[#oleResV:]] FPFastMathMode NSZ
; CHECK: OpDecorate %[[#ogeResV:]] FPFastMathMode AllowRecip
; CHECK: OpDecorate %[[#ordResV:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#ueqResV:]] FPFastMathMode NotNaN|NotInf
; CHECK: OpDecorate %[[#maxResV:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: OpDecorate %[[#maxCommonResV:]] FPFastMathMode NotNaN|NotInf

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
  store volatile float %addRes, ptr @G_addRes
  ; CHECK: %[[#subRes]] = OpFSub
  %subRes = fsub nnan float %1,  %2
  store volatile float %subRes, ptr @G_subRes
  ; CHECK: %[[#mulRes]] = OpFMul
  %mulRes = fmul ninf float %1,  %2
  store volatile float %mulRes, ptr @G_mulRes
  ; CHECK: %[[#divRes]] = OpFDiv
  %divRes = fdiv nsz float %1,  %2
  store volatile float %divRes, ptr @G_divRes
  ; CHECK: %[[#remRes]] = OpFRem
  %remRes = frem arcp float %1,  %2
  store volatile float %remRes, ptr @G_remRes
  ; CHECK: %[[#negRes]] = OpFNegate
  %negRes = fneg fast float %1
  store volatile float %negRes, ptr @G_negRes
  ; CHECK: %[[#oeqRes]] = OpFOrdEqual
  %oeqRes = fcmp nnan ninf oeq float %1,  %2
  store volatile i1 %oeqRes, ptr @G_oeqRes
  %oneRes = fcmp one float %1,  %2, !spirv.Decorations !3
  store volatile i1 %oneRes, ptr @G_oneRes
  ; CHECK: %[[#oltRes]] = OpFOrdLessThan
  %oltRes = fcmp nnan olt float %1,  %2, !spirv.Decorations !3
  store volatile i1 %oltRes, ptr @G_oltRes
  ; CHECK: %[[#ogtRes]] = OpFOrdGreaterThan
  %ogtRes = fcmp ninf ogt float %1,  %2, !spirv.Decorations !3
  store volatile i1 %ogtRes, ptr @G_ogtRes
  ; CHECK: %[[#oleRes]] = OpFOrdLessThanEqual
  %oleRes = fcmp nsz ole float %1,  %2, !spirv.Decorations !3
  store volatile i1 %oleRes, ptr @G_oleRes
  ; CHECK: %[[#ogeRes]] = OpFOrdGreaterThanEqual
  %ogeRes = fcmp arcp oge float %1,  %2, !spirv.Decorations !3
  store volatile i1 %ogeRes, ptr @G_ogeRes
  ; CHECK: %[[#ordRes]] = OpOrdered
  %ordRes = fcmp fast ord float %1,  %2, !spirv.Decorations !3
  store volatile i1 %ordRes, ptr @G_ordRes
  ; CHECK: %[[#ueqRes]] = OpFUnordEqual
  %ueqRes = fcmp nnan ninf ueq float %1,  %2, !spirv.Decorations !3
  store volatile i1 %ueqRes, ptr @G_ueqRes
  %uneRes = fcmp une float %1,  %2, !spirv.Decorations !3
  store volatile i1 %uneRes, ptr @G_uneRes
  %ultRes = fcmp ult float %1,  %2, !spirv.Decorations !3
  store volatile i1 %ultRes, ptr @G_ultRes
  %ugtRes = fcmp ugt float %1,  %2, !spirv.Decorations !3
  store volatile i1 %ugtRes, ptr @G_ugtRes
  %uleRes = fcmp ule float %1,  %2, !spirv.Decorations !3
  store volatile i1 %uleRes, ptr @G_uleRes
  %ugeRes = fcmp uge float %1,  %2, !spirv.Decorations !3
  store volatile i1 %ugeRes, ptr @G_ugeRes
  %unoRes = fcmp uno float %1,  %2, !spirv.Decorations !3
  store volatile i1 %unoRes, ptr @G_unoRes
  %modRes = call spir_func float @_Z4fmodff(float %1, float %2)
  store volatile float %modRes, ptr @G_modRes
  ; CHECK: %[[#maxRes]] = OpExtInst %[[#]] %[[#]] fmax
  %maxRes = tail call fast spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  store volatile float %maxRes, ptr @G_maxRes
   ; CHECK: %[[#maxCommonRes]] = OpExtInst %[[#]] %[[#]] fmax
   %maxCommonRes = tail call spir_func noundef float @_Z23__spirv_ocl_fmax_commonff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  store volatile float %maxCommonRes, ptr @G_maxCommonRes
  ret void
}

define weak_odr dso_local spir_kernel void @fooV(<2 x float> %v1, <2 x float> %v2) {
  %addResV = fadd <2 x float> %v1,  %v2
  store volatile <2 x float> %addResV, ptr @G_addResV
  %subResV = fsub nnan <2 x float> %v1,  %v2
  store volatile <2 x float> %subResV, ptr @G_subResV
  %mulResV = fmul ninf <2 x float> %v1,  %v2
  store volatile <2 x float> %mulResV, ptr @G_mulResV
  %divResV = fdiv nsz <2 x float> %v1,  %v2
  store volatile <2 x float> %divResV, ptr @G_divResV
  %remResV = frem arcp <2 x float> %v1,  %v2
  store volatile <2 x float> %remResV, ptr @G_remResV
  %negResV = fneg fast <2 x float> %v1
  store volatile <2 x float> %negResV, ptr @G_negResV
  %oeqResV = fcmp nnan ninf oeq <2 x float> %v1,  %v2
  store volatile <2 x i1> %oeqResV, ptr @G_oeqResV
  %oneResV = fcmp one <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %oneResV, ptr @G_oneResV
  %oltResV = fcmp nnan olt <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %oltResV, ptr @G_oltResV
  %ogtResV = fcmp ninf ogt <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %ogtResV, ptr @G_ogtResV
  %oleResV = fcmp nsz ole <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %oleResV, ptr @G_oleResV
  %ogeResV = fcmp arcp oge <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %ogeResV, ptr @G_ogeResV
  %ordResV = fcmp fast ord <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %ordResV, ptr @G_ordResV
  %ueqResV = fcmp nnan ninf ueq <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %ueqResV, ptr @G_ueqResV
  %uneResV = fcmp une <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %uneResV, ptr @G_uneResV
  %ultResV = fcmp ult <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %ultResV, ptr @G_ultResV
  %ugtResV = fcmp ugt <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %ugtResV, ptr @G_ugtResV
  %uleResV = fcmp ule <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %uleResV, ptr @G_uleResV
  %ugeResV = fcmp uge <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %ugeResV, ptr @G_ugeResV
  %unoResV = fcmp uno <2 x float> %v1,  %v2, !spirv.Decorations !3
  store volatile <2 x i1> %unoResV, ptr @G_unoResV
  %modResV = call spir_func <2 x float> @_Z4fmodDv2_fDv2_f(<2 x float> %v1, <2 x float> %v2)
  store volatile <2 x float> %modResV, ptr @G_modResV
  %maxResV = tail call fast spir_func noundef nofpclass(nan inf) <2 x float> @_Z16__spirv_ocl_fmaxDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
  store volatile <2 x float> %maxResV, ptr @G_maxResV
   %maxCommonResV = tail call spir_func noundef <2 x float> @_Z23__spirv_ocl_fmax_commonDv2_fDv2_f(<2 x float> noundef nofpclass(nan inf) %v1, <2 x float> noundef nofpclass(nan inf) %v2)
  store volatile <2 x float> %maxCommonResV, ptr @G_maxCommonResV
  ret void
}

!3 = !{!5, !4}
!4 = !{i32 42} ; 42 is NoContraction decoration
!5 = !{i32 40, i32 393216} ; 40 is FPFastMathMode
