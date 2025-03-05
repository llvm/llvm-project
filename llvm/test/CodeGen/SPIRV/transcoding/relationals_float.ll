; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; This test checks following SYCL relational builtins with float and float2
;; types:
;;   isfinite, isinf, isnan, isnormal, signbit, isequal, isnotequal, isgreater
;;   isgreaterequal, isless, islessequal, islessgreater, isordered, isunordered

; CHECK-SPIRV: %[[#BoolTypeID:]] = OpTypeBool
; CHECK-SPIRV: %[[#BoolVectorTypeID:]] = OpTypeVector %[[#BoolTypeID]] 2

; CHECK-SPIRV: OpIsFinite %[[#BoolTypeID]]
; CHECK-SPIRV: OpIsInf %[[#BoolTypeID]]
; CHECK-SPIRV: OpIsNan %[[#BoolTypeID]]
; CHECK-SPIRV: OpIsNormal %[[#BoolTypeID]]
; CHECK-SPIRV: OpSignBitSet %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpFUnordNotEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdGreaterThan %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdGreaterThanEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdLessThan %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdLessThanEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdNotEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpOrdered %[[#BoolTypeID]]
; CHECK-SPIRV: OpUnordered %[[#BoolTypeID]]

; CHECK-SPIRV: OpIsFinite %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpIsInf %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpIsNan %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpIsNormal %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpSignBitSet %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFUnordNotEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdGreaterThan %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdGreaterThanEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdLessThan %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdLessThanEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdNotEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpOrdered %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpUnordered %[[#BoolVectorTypeID]]

define dso_local spir_func void @test_scalar(i32 addrspace(4)* nocapture writeonly %out, float %f) local_unnamed_addr {
entry:
  %call = tail call spir_func i32 @_Z8isfinitef(float %f)
  %call1 = tail call spir_func i32 @_Z5isinff(float %f)
  %add = add nsw i32 %call1, %call
  %call2 = tail call spir_func i32 @_Z5isnanf(float %f)
  %add3 = add nsw i32 %add, %call2
  %call4 = tail call spir_func i32 @_Z8isnormalf(float %f)
  %add5 = add nsw i32 %add3, %call4
  %call6 = tail call spir_func i32 @_Z7signbitf(float %f)
  %add7 = add nsw i32 %add5, %call6
  %call8 = tail call spir_func i32 @_Z7isequalff(float %f, float %f)
  %add9 = add nsw i32 %add7, %call8
  %call10 = tail call spir_func i32 @_Z10isnotequalff(float %f, float %f)
  %add11 = add nsw i32 %add9, %call10
  %call12 = tail call spir_func i32 @_Z9isgreaterff(float %f, float %f)
  %add13 = add nsw i32 %add11, %call12
  %call14 = tail call spir_func i32 @_Z14isgreaterequalff(float %f, float %f)
  %add15 = add nsw i32 %add13, %call14
  %call16 = tail call spir_func i32 @_Z6islessff(float %f, float %f)
  %add17 = add nsw i32 %add15, %call16
  %call18 = tail call spir_func i32 @_Z11islessequalff(float %f, float %f)
  %add19 = add nsw i32 %add17, %call18
  %call20 = tail call spir_func i32 @_Z13islessgreaterff(float %f, float %f)
  %add21 = add nsw i32 %add19, %call20
  %call22 = tail call spir_func i32 @_Z9isorderedff(float %f, float %f)
  %add23 = add nsw i32 %add21, %call22
  %call24 = tail call spir_func i32 @_Z11isunorderedff(float %f, float %f)
  %add25 = add nsw i32 %add23, %call24
  store i32 %add25, i32 addrspace(4)* %out, align 4
  ret void
}

declare spir_func i32 @_Z8isfinitef(float) local_unnamed_addr

declare spir_func i32 @_Z5isinff(float) local_unnamed_addr

declare spir_func i32 @_Z5isnanf(float) local_unnamed_addr

declare spir_func i32 @_Z8isnormalf(float) local_unnamed_addr

declare spir_func i32 @_Z7signbitf(float) local_unnamed_addr

declare spir_func i32 @_Z7isequalff(float, float) local_unnamed_addr

declare spir_func i32 @_Z10isnotequalff(float, float) local_unnamed_addr

declare spir_func i32 @_Z9isgreaterff(float, float) local_unnamed_addr

declare spir_func i32 @_Z14isgreaterequalff(float, float) local_unnamed_addr

declare spir_func i32 @_Z6islessff(float, float) local_unnamed_addr

declare spir_func i32 @_Z11islessequalff(float, float) local_unnamed_addr

declare spir_func i32 @_Z13islessgreaterff(float, float) local_unnamed_addr

declare spir_func i32 @_Z9isorderedff(float, float) local_unnamed_addr

declare spir_func i32 @_Z11isunorderedff(float, float) local_unnamed_addr

define dso_local spir_func void @test_vector(<2 x i32> addrspace(4)* nocapture writeonly %out, <2 x float> %f) local_unnamed_addr {
entry:
  %call = tail call spir_func <2 x i32> @_Z8isfiniteDv2_f(<2 x float> %f)
  %call1 = tail call spir_func <2 x i32> @_Z5isinfDv2_f(<2 x float> %f)
  %add = add <2 x i32> %call1, %call
  %call2 = tail call spir_func <2 x i32> @_Z5isnanDv2_f(<2 x float> %f)
  %add3 = add <2 x i32> %add, %call2
  %call4 = tail call spir_func <2 x i32> @_Z8isnormalDv2_f(<2 x float> %f)
  %add5 = add <2 x i32> %add3, %call4
  %call6 = tail call spir_func <2 x i32> @_Z7signbitDv2_f(<2 x float> %f)
  %add7 = add <2 x i32> %add5, %call6
  %call8 = tail call spir_func <2 x i32> @_Z7isequalDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add9 = add <2 x i32> %add7, %call8
  %call10 = tail call spir_func <2 x i32> @_Z10isnotequalDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add11 = add <2 x i32> %add9, %call10
  %call12 = tail call spir_func <2 x i32> @_Z9isgreaterDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add13 = add <2 x i32> %add11, %call12
  %call14 = tail call spir_func <2 x i32> @_Z14isgreaterequalDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add15 = add <2 x i32> %add13, %call14
  %call16 = tail call spir_func <2 x i32> @_Z6islessDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add17 = add <2 x i32> %add15, %call16
  %call18 = tail call spir_func <2 x i32> @_Z11islessequalDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add19 = add <2 x i32> %add17, %call18
  %call20 = tail call spir_func <2 x i32> @_Z13islessgreaterDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add21 = add <2 x i32> %add19, %call20
  %call22 = tail call spir_func <2 x i32> @_Z9isorderedDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add23 = add <2 x i32> %add21, %call22
  %call24 = tail call spir_func <2 x i32> @_Z11isunorderedDv2_fS_(<2 x float> %f, <2 x float> %f)
  %add25 = add <2 x i32> %add23, %call24
  store <2 x i32> %add25, <2 x i32> addrspace(4)* %out, align 8
  ret void
}

declare spir_func <2 x i32> @_Z8isfiniteDv2_f(<2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z5isinfDv2_f(<2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z5isnanDv2_f(<2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z8isnormalDv2_f(<2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z7signbitDv2_f(<2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z7isequalDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z10isnotequalDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z9isgreaterDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z14isgreaterequalDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z6islessDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z11islessequalDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z13islessgreaterDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z9isorderedDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr

declare spir_func <2 x i32> @_Z11isunorderedDv2_fS_(<2 x float>, <2 x float>) local_unnamed_addr
