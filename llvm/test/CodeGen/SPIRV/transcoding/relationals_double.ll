; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; This test checks following SYCL relational builtins with double and double2
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

define dso_local spir_func void @test_scalar(i32 addrspace(4)* nocapture writeonly %out, double %d) local_unnamed_addr {
entry:
  %call = tail call spir_func i32 @_Z8isfinited(double %d)
  %call1 = tail call spir_func i32 @_Z5isinfd(double %d)
  %add = add nsw i32 %call1, %call
  %call2 = tail call spir_func i32 @_Z5isnand(double %d)
  %add3 = add nsw i32 %add, %call2
  %call4 = tail call spir_func i32 @_Z8isnormald(double %d)
  %add5 = add nsw i32 %add3, %call4
  %call6 = tail call spir_func i32 @_Z7signbitd(double %d)
  %add7 = add nsw i32 %add5, %call6
  %call8 = tail call spir_func i32 @_Z7isequaldd(double %d, double %d)
  %add9 = add nsw i32 %add7, %call8
  %call10 = tail call spir_func i32 @_Z10isnotequaldd(double %d, double %d)
  %add11 = add nsw i32 %add9, %call10
  %call12 = tail call spir_func i32 @_Z9isgreaterdd(double %d, double %d)
  %add13 = add nsw i32 %add11, %call12
  %call14 = tail call spir_func i32 @_Z14isgreaterequaldd(double %d, double %d)
  %add15 = add nsw i32 %add13, %call14
  %call16 = tail call spir_func i32 @_Z6islessdd(double %d, double %d)
  %add17 = add nsw i32 %add15, %call16
  %call18 = tail call spir_func i32 @_Z11islessequaldd(double %d, double %d)
  %add19 = add nsw i32 %add17, %call18
  %call20 = tail call spir_func i32 @_Z13islessgreaterdd(double %d, double %d)
  %add21 = add nsw i32 %add19, %call20
  %call22 = tail call spir_func i32 @_Z9isordereddd(double %d, double %d)
  %add23 = add nsw i32 %add21, %call22
  %call24 = tail call spir_func i32 @_Z11isunordereddd(double %d, double %d)
  %add25 = add nsw i32 %add23, %call24
  store i32 %add25, i32 addrspace(4)* %out, align 4
  ret void
}

declare spir_func i32 @_Z8isfinited(double) local_unnamed_addr

declare spir_func i32 @_Z5isinfd(double) local_unnamed_addr

declare spir_func i32 @_Z5isnand(double) local_unnamed_addr

declare spir_func i32 @_Z8isnormald(double) local_unnamed_addr

declare spir_func i32 @_Z7signbitd(double) local_unnamed_addr

declare spir_func i32 @_Z7isequaldd(double, double) local_unnamed_addr

declare spir_func i32 @_Z10isnotequaldd(double, double) local_unnamed_addr

declare spir_func i32 @_Z9isgreaterdd(double, double) local_unnamed_addr

declare spir_func i32 @_Z14isgreaterequaldd(double, double) local_unnamed_addr

declare spir_func i32 @_Z6islessdd(double, double) local_unnamed_addr

declare spir_func i32 @_Z11islessequaldd(double, double) local_unnamed_addr

declare spir_func i32 @_Z13islessgreaterdd(double, double) local_unnamed_addr

declare spir_func i32 @_Z9isordereddd(double, double) local_unnamed_addr

declare spir_func i32 @_Z11isunordereddd(double, double) local_unnamed_addr

define dso_local spir_func void @test_vector(<2 x i64> addrspace(4)* nocapture writeonly %out, <2 x double> %d) local_unnamed_addr {
entry:
  %call = tail call spir_func <2 x i64> @_Z8isfiniteDv2_d(<2 x double> %d)
  %call1 = tail call spir_func <2 x i64> @_Z5isinfDv2_d(<2 x double> %d)
  %add = add <2 x i64> %call1, %call
  %call2 = tail call spir_func <2 x i64> @_Z5isnanDv2_d(<2 x double> %d)
  %add3 = add <2 x i64> %add, %call2
  %call4 = tail call spir_func <2 x i64> @_Z8isnormalDv2_d(<2 x double> %d)
  %add5 = add <2 x i64> %add3, %call4
  %call6 = tail call spir_func <2 x i64> @_Z7signbitDv2_d(<2 x double> %d)
  %add7 = add <2 x i64> %add5, %call6
  %call8 = tail call spir_func <2 x i64> @_Z7isequalDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add9 = add <2 x i64> %add7, %call8
  %call10 = tail call spir_func <2 x i64> @_Z10isnotequalDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add11 = add <2 x i64> %add9, %call10
  %call12 = tail call spir_func <2 x i64> @_Z9isgreaterDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add13 = add <2 x i64> %add11, %call12
  %call14 = tail call spir_func <2 x i64> @_Z14isgreaterequalDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add15 = add <2 x i64> %add13, %call14
  %call16 = tail call spir_func <2 x i64> @_Z6islessDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add17 = add <2 x i64> %add15, %call16
  %call18 = tail call spir_func <2 x i64> @_Z11islessequalDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add19 = add <2 x i64> %add17, %call18
  %call20 = tail call spir_func <2 x i64> @_Z13islessgreaterDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add21 = add <2 x i64> %add19, %call20
  %call22 = tail call spir_func <2 x i64> @_Z9isorderedDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add23 = add <2 x i64> %add21, %call22
  %call24 = tail call spir_func <2 x i64> @_Z11isunorderedDv2_dS_(<2 x double> %d, <2 x double> %d)
  %add25 = add <2 x i64> %add23, %call24
  store <2 x i64> %add25, <2 x i64> addrspace(4)* %out, align 16
  ret void
}

declare spir_func <2 x i64> @_Z8isfiniteDv2_d(<2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z5isinfDv2_d(<2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z5isnanDv2_d(<2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z8isnormalDv2_d(<2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z7signbitDv2_d(<2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z7isequalDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z10isnotequalDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z9isgreaterDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z14isgreaterequalDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z6islessDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z11islessequalDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z13islessgreaterDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z9isorderedDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr

declare spir_func <2 x i64> @_Z11isunorderedDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr
