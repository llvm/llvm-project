
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s 

; CHECK: OpCapability Kernel
; CHECK: OpCapability ArbitraryPrecisionFloatingPointINTEL
; CHECK: OpCapability ArbitraryPrecisionIntegersINTEL
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_floating_point"
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_integers"

; CHECK-DAG: %[[Ty_2:[0-9]+]] = OpTypeInt 2 0
; CHECK-DAG: %[[Ty_5:[0-9]+]] = OpTypeInt 5 0
; CHECK-DAG: %[[Ty_7:[0-9]+]] = OpTypeInt 7 0
; CHECK-DAG: %[[Ty_9:[0-9]+]] = OpTypeInt 9 0
; CHECK-DAG: %[[Ty_11:[0-9]+]] = OpTypeInt 11 0
; CHECK-DAG: %[[Ty_13:[0-9]+]] = OpTypeInt 13 0
; CHECK-DAG: %[[Ty_14:[0-9]+]] = OpTypeInt 14 0
; CHECK-DAG: %[[Ty_15:[0-9]+]] = OpTypeInt 15 0
; CHECK-DAG: %[[Ty_16:[0-9]+]] = OpTypeInt 16 0
; CHECK-DAG: %[[Ty_17:[0-9]+]] = OpTypeInt 17 0
; CHECK-DAG: %[[Ty_18:[0-9]+]] = OpTypeInt 18 0
; CHECK-DAG: %[[Ty_20:[0-9]+]] = OpTypeInt 20 0
; CHECK-DAG: %[[Ty_23:[0-9]+]] = OpTypeInt 23 0
; CHECK-DAG: %[[Ty_25:[0-9]+]] = OpTypeInt 25 0
; CHECK-DAG: %[[Ty_30:[0-9]+]] = OpTypeInt 30 0
; CHECK-DAG: %[[Ty_32:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[Ty_34:[0-9]+]] = OpTypeInt 34 0
; CHECK-DAG: %[[Ty_39:[0-9]+]] = OpTypeInt 39 0
; CHECK-DAG: %[[Ty_40:[0-9]+]] = OpTypeInt 40 0
; CHECK-DAG: %[[Ty_41:[0-9]+]] = OpTypeInt 41 0
; CHECK-DAG: %[[Ty_42:[0-9]+]] = OpTypeInt 42 0
; CHECK-DAG: %[[Ty_43:[0-9]+]] = OpTypeInt 43 0
; CHECK-DAG: %[[Ty_47:[0-9]+]] = OpTypeInt 47 0
; CHECK-DAG: %[[Ty_51:[0-9]+]] = OpTypeInt 51 0
; CHECK-DAG: %[[Ty_55:[0-9]+]] = OpTypeInt 55 0
; CHECK-DAG: %[[Ty_63:[0-9]+]] = OpTypeInt 63 0
; CHECK-DAG: %[[Ty_Bool:[0-9]+]] = OpTypeBool

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

; Function Attrs: norecurse
define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
  %1 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  %2 = addrspacecast ptr %1 to ptr addrspace(4)
  call spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %2)
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: inlinehint norecurse
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %0) #2 align 2 {
  %2 = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %0, ptr %2, align 8, !tbaa !5
  call spir_func void @_Z13ap_float_castILi11ELi28ELi9ELi30EEvv()
  call spir_func void @_Z22ap_float_cast_from_intILi43ELi8ELi16EEvv()
  call spir_func void @_Z20ap_float_cast_to_intILi7ELi15ELi30EEvv()
  call spir_func void @_Z12ap_float_addILi5ELi7ELi6ELi8ELi4ELi9EEvv()
  call spir_func void @_Z12ap_float_addILi6ELi8ELi4ELi9ELi5ELi7EEvv()
  call spir_func void @_Z12ap_float_subILi4ELi4ELi5ELi5ELi6ELi6EEvv()
  call spir_func void @_Z12ap_float_mulILi16ELi34ELi16ELi34ELi16ELi34EEvv()
  call spir_func void @_Z12ap_float_divILi4ELi11ELi4ELi11ELi5ELi12EEvv()
  call spir_func void @_Z11ap_float_gtILi20ELi42ELi21ELi41EEvv()
  call spir_func void @_Z11ap_float_geILi19ELi27ELi19ELi27EEvv()
  call spir_func void @_Z11ap_float_ltILi2ELi2ELi3ELi3EEvv()
  call spir_func void @_Z11ap_float_leILi27ELi27ELi26ELi28EEvv()
  call spir_func void @_Z11ap_float_eqILi7ELi12ELi7ELi7EEvv()
  call spir_func void @_Z14ap_float_recipILi9ELi29ELi9ELi29EEvv()
  call spir_func void @_Z14ap_float_rsqrtILi12ELi19ELi13ELi20EEvv()
  call spir_func void @_Z13ap_float_cbrtILi0ELi1ELi0ELi1EEvv()
  call spir_func void @_Z14ap_float_hypotILi20ELi20ELi21ELi21ELi19ELi22EEvv()
  call spir_func void @_Z13ap_float_sqrtILi7ELi7ELi8ELi8EEvv()
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_castILi11ELi28ELi9ELi30EEvv() #3 {
  %1 = alloca i40, align 8
  %2 = alloca i40, align 8
  %3 = load i40, ptr %1, align 8, !tbaa !9
  %4 = call spir_func i40 @_Z31__spirv_ArbitraryFloatCastINTELILi40ELi40EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i40 %3, i32 28, i32 30, i32 0, i32 2, i32 1) #5
; CHECK: %[[LoadVar:[0-9]+]] = OpLoad %[[Ty_40]] %[[SourceVar:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[CastResult:[0-9]+]] = OpArbitraryFloatCastINTEL %[[Ty_40]] %[[LoadVar]] 28 30 0 2 1
  store i40 %4, ptr %2, align 8, !tbaa !9
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z22ap_float_cast_from_intILi43ELi8ELi16EEvv() #3 {
  %1 = alloca i43, align 8
  %2 = alloca i25, align 4
  %3 = load i43, ptr %1, align 8, !tbaa !11
  %4 = call spir_func signext i25 @_Z38__spirv_ArbitraryFloatCastFromIntINTELILi43ELi25EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiibiii(i43 %3, i32 16, i1 zeroext false, i32 0, i32 2, i32 1) #5
; CHECK: %[[LoadVar:[0-9]+]] = OpLoad %[[Ty_43]] %[[SourceVar:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[CastResult:[0-9]+]] = OpArbitraryFloatCastFromIntINTEL %[[Ty_25]] %[[LoadVar]] 16 0 0 2 1
  store i25 %4, ptr %2, align 4, !tbaa !13
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z20ap_float_cast_to_intILi7ELi15ELi30EEvv() #3 {
  %1 = alloca i23, align 4
  %2 = alloca i30, align 4
  %3 = load i23, ptr %1, align 4, !tbaa !15
  %4 = call spir_func signext i30 @_Z36__spirv_ArbitraryFloatCastToIntINTELILi23ELi30EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiibiii(i23 signext %3, i32 15, i1 zeroext true, i32 0, i32 2, i32 1) #5
; CHECK: %[[LoadVar:[0-9]+]] = OpLoad %[[Ty_23]] %[[SourceVar:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[CastResult:[0-9]+]] = OpArbitraryFloatCastToIntINTEL %[[Ty_30]] %[[LoadVar]] 15 1 0 2 1
  store i30 %4, ptr %2, align 4, !tbaa !17
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_addILi5ELi7ELi6ELi8ELi4ELi9EEvv() #3 {
  %1 = alloca i13, align 2
  %2 = alloca i13, align 2
  %3 = alloca i15, align 2
  %4 = alloca i15, align 2
  %5 = alloca i14, align 2
  %6 = alloca i14, align 2
  %7 = load i13, ptr %1, align 2, !tbaa !19
  %8 = load i15, ptr %3, align 2, !tbaa !21
  %9 = call spir_func signext i14 @_Z30__spirv_ArbitraryFloatAddINTELILi13ELi15ELi14EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i13 signext %7, i32 7, i15 signext %8, i32 8, i32 9, i32 0, i32 2, i32 1) #5
; CHECK: %[[AddOp1:[0-9]+]] = OpLoad %[[Ty_13]] %[[Src1:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[AddOp2:[0-9]+]] = OpLoad %[[Ty_15]] %[[Src2:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[AddResult:[0-9]+]] = OpArbitraryFloatAddINTEL %[[Ty_14]] %[[AddOp1]] 7 %[[AddOp2]] 8 9 0 2 1
  store i14 %9, ptr %5, align 2, !tbaa !23
  call void @llvm.lifetime.start.p0(i64 2, ptr %6) #5
  %10 = load i13, ptr %2, align 2, !tbaa !19
  %11 = load i15, ptr %4, align 2, !tbaa !21
  %12 = call spir_func signext i14 @_Z30__spirv_ArbitraryFloatAddINTELILi13ELi15ELi14EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i13 signext %10, i32 7, i15 signext %11, i32 8, i32 9, i32 0, i32 2, i32 1) #5
; CHECK: %[[AddOp1:[0-9]+]] = OpLoad %[[Ty_13]] %[[Src1:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[AddOp2:[0-9]+]] = OpLoad %[[Ty_15]] %[[Src2:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[AddResult:[0-9]+]] = OpArbitraryFloatAddINTEL %[[Ty_14]] %[[AddOp1]] 7 %[[AddOp2]] 8 9 0 2 1
  store i14 %12, ptr %6, align 2, !tbaa !23
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_addILi6ELi8ELi4ELi9ELi5ELi7EEvv() #3 {
  %1 = alloca i15, align 2
  %2 = alloca i15, align 2
  %3 = alloca i14, align 2
  %4 = alloca i14, align 2
  %5 = alloca i13, align 2
  %6 = alloca i13, align 2
  %7 = load i15, ptr %1, align 2, !tbaa !21
  %8 = load i14, ptr %3, align 2, !tbaa !23
  %9 = call spir_func signext i13 @_Z30__spirv_ArbitraryFloatAddINTELILi15ELi14ELi13EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i15 signext %7, i32 8, i14 signext %8, i32 9, i32 7, i32 0, i32 2, i32 1) #5
; CHECK: %[[AddOp1:[0-9]+]] = OpLoad %[[Ty_15]] %[[Src1:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[AddOp2:[0-9]+]] = OpLoad %[[Ty_14]] %[[Src2:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[AddResult:[0-9]+]] = OpArbitraryFloatAddINTEL %[[Ty_13]] %[[AddOp1]] 8 %[[AddOp2]] 9 7 0 2 1
  store i13 %9, ptr %5, align 2, !tbaa !19
  call void @llvm.lifetime.start.p0(i64 2, ptr %6) #5
  %10 = load i15, ptr %2, align 2, !tbaa !21
  %11 = load i14, ptr %4, align 2, !tbaa !23
  %12 = call spir_func signext i13 @_Z30__spirv_ArbitraryFloatAddINTELILi15ELi14ELi13EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i15 signext %10, i32 8, i14 signext %11, i32 9, i32 7, i32 0, i32 2, i32 1) #5
; CHECK: %[[AddOp1:[0-9]+]] = OpLoad %[[Ty_15]] %[[Add2_A2:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[AddOp2:[0-9]+]] = OpLoad %[[Ty_14]] %[[Add2_B2:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[AddResult:[0-9]+]] = OpArbitraryFloatAddINTEL %[[Ty_13]] %[[AddOp1]] 8 %[[AddOp2]] 9 7 0 2 1
  store i13 %12, ptr %6, align 2, !tbaa !19
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_subILi4ELi4ELi5ELi5ELi6ELi6EEvv() #3 {
  %1 = alloca i9, align 2
  %2 = alloca i11, align 2
  %3 = alloca i13, align 2
  %4 = load i9, ptr %1, align 2, !tbaa !25
  %5 = load i11, ptr %2, align 2, !tbaa !27
  %6 = call spir_func signext i13 @_Z30__spirv_ArbitraryFloatSubINTELILi9ELi11ELi13EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i9 signext %4, i32 4, i11 signext %5, i32 5, i32 6, i32 0, i32 2, i32 1) #5
; CHECK: %[[SubOp1:[0-9]+]] = OpLoad %[[Ty_9]] %[[Sub_A:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[SubOp2:[0-9]+]] = OpLoad %[[Ty_11]] %[[Sub_B:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[SubResult:[0-9]+]] = OpArbitraryFloatSubINTEL %[[Ty_13]] %[[SubOp1]] 4 %[[SubOp2]] 5 6 0 2 1
  store i13 %6, ptr %3, align 2, !tbaa !19
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_mulILi16ELi34ELi16ELi34ELi16ELi34EEvv() #3 {
  %1 = alloca i51, align 8
  %2 = alloca i51, align 8
  %3 = alloca i51, align 8
  %4 = load i51, ptr %1, align 8, !tbaa !29
  %5 = load i51, ptr %2, align 8, !tbaa !29
  %6 = call spir_func i51 @_Z30__spirv_ArbitraryFloatMulINTELILi51ELi51ELi51EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i51 %4, i32 34, i51 %5, i32 34, i32 34, i32 0, i32 2, i32 1) #5
; CHECK: %[[MulOp1:[0-9]+]] = OpLoad %[[Ty_51]] %[[Mul_A:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[MulOp2:[0-9]+]] = OpLoad %[[Ty_51]] %[[Mul_B:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[MulResult:[0-9]+]] = OpArbitraryFloatMulINTEL %[[Ty_51]] %[[MulOp1]] 34 %[[MulOp2]] 34 34 0 2 1
  store i51 %6, ptr %3, align 8, !tbaa !29
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_divILi4ELi11ELi4ELi11ELi5ELi12EEvv() #3 {
  %1 = alloca i16, align 2
  %2 = alloca i16, align 2
  %3 = alloca i18, align 4
  %4 = load i16, ptr %1, align 2, !tbaa !31
  %5 = load i16, ptr %2, align 2, !tbaa !31
  %6 = call spir_func signext i18 @_Z30__spirv_ArbitraryFloatDivINTELILi16ELi16ELi18EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i16 signext %4, i32 11, i16 signext %5, i32 11, i32 12, i32 0, i32 2, i32 1) #5
; CHECK: %[[DivOp1:[0-9]+]] = OpLoad %[[Ty_16]] %[[Div_A:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[DivOp2:[0-9]+]] = OpLoad %[[Ty_16]] %[[Div_B:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[DivResult:[0-9]+]] = OpArbitraryFloatDivINTEL %[[Ty_18]] %[[DivOp1]] 11 %[[DivOp2]] 11 12 0 2 1
  store i18 %6, ptr %3, align 4, !tbaa !33
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z11ap_float_gtILi20ELi42ELi21ELi41EEvv() #3 {
  %1 = alloca i63, align 8
  %2 = alloca i63, align 8
  %3 = alloca i8, align 1
  %4 = load i63, ptr %1, align 8, !tbaa !35
  %5 = load i63, ptr %2, align 8, !tbaa !35
  %6 = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatGTINTELILi63ELi63EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i63 %4, i32 42, i63 %5, i32 41) #5
; CHECK: %[[GTOp1:[0-9]+]] = OpLoad %[[Ty_63]] %[[GT_A:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[GTOp2:[0-9]+]] = OpLoad %[[Ty_63]] %[[GT_B:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[GTResult:[0-9]+]] = OpArbitraryFloatGTINTEL %[[Ty_Bool]] %[[GTOp1]] 42 %[[GTOp2]] 41
  %7 = zext i1 %6 to i8
  store i8 %7, ptr %3, align 1, !tbaa !37
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z11ap_float_geILi19ELi27ELi19ELi27EEvv() #3 {
  %1 = alloca i47, align 8
  %2 = alloca i47, align 8
  %3 = alloca i8, align 1
  %4 = load i47, ptr %1, align 8, !tbaa !39
  %5 = load i47, ptr %2, align 8, !tbaa !39
  %6 = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatGEINTELILi47ELi47EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i47 %4, i32 27, i47 %5, i32 27) #5
; CHECK: %[[GE_A1:[0-9]+]] = OpLoad %[[Ty_47]] %[[GE_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[GE_B1:[0-9]+]] = OpLoad %[[Ty_47]] %[[GE_BId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[GEResult:[0-9]+]] = OpArbitraryFloatGEINTEL %[[Ty_Bool]] %[[GE_A1]] 27 %[[GE_B1]] 27
  %7 = zext i1 %6 to i8
  store i8 %7, ptr %3, align 1, !tbaa !37
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z11ap_float_ltILi2ELi2ELi3ELi3EEvv() #3 {
  %1 = alloca i5, align 1
  %2 = alloca i7, align 1
  %3 = alloca i8, align 1
  %4 = load i5, ptr %1, align 1, !tbaa !41
  %5 = load i7, ptr %2, align 1, !tbaa !43
  %6 = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatLTINTELILi5ELi7EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i5 signext %4, i32 2, i7 signext %5, i32 3) #5
; CHECK: %[[LT_A1:[0-9]+]] = OpLoad %[[Ty_5]] %[[LT_AId:[0-9]+]] Aligned 1
; CHECK-NEXT: %[[LT_B1:[0-9]+]] = OpLoad %[[Ty_7]] %[[LT_BId:[0-9]+]] Aligned 1
; CHECK-NEXT: %[[LTResult:[0-9]+]] = OpArbitraryFloatLTINTEL %[[Ty_Bool]] %[[LT_A1]] 2 %[[LT_B1]] 3
  %7 = zext i1 %6 to i8
  store i8 %7, ptr %3, align 1, !tbaa !37
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z11ap_float_leILi27ELi27ELi26ELi28EEvv() #3 {
  %1 = alloca i55, align 8
  %2 = alloca i55, align 8
  %3 = alloca i8, align 1
  %4 = load i55, ptr %1, align 8, !tbaa !45
  %5 = load i55, ptr %2, align 8, !tbaa !45
  %6 = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatLEINTELILi55ELi55EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i55 %4, i32 27, i55 %5, i32 28) #5
; CHECK: %[[LE_A1:[0-9]+]] = OpLoad %[[Ty_55]] %[[LE_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[LE_B1:[0-9]+]] = OpLoad %[[Ty_55]] %[[LE_BId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[LEResult:[0-9]+]] = OpArbitraryFloatLEINTEL %[[Ty_Bool]] %[[LE_A1]] 27 %[[LE_B1]] 28
  %7 = zext i1 %6 to i8
  store i8 %7, ptr %3, align 1, !tbaa !37
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z11ap_float_eqILi7ELi12ELi7ELi7EEvv() #3 {
  %1 = alloca i20, align 4
  %2 = alloca i15, align 2
  %3 = alloca i8, align 1
  %4 = load i20, ptr %1, align 4, !tbaa !47
  %5 = load i15, ptr %2, align 2, !tbaa !21
  %6 = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatEQINTELILi20ELi15EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i20 signext %4, i32 12, i15 signext %5, i32 7) #5
; CHECK: %[[EQ_A1:[0-9]+]] = OpLoad %[[Ty_20]] %[[EQ_AId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[EQ_B1:[0-9]+]] = OpLoad %[[Ty_15]] %[[EQ_BId:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[EQResult:[0-9]+]] = OpArbitraryFloatEQINTEL %[[Ty_Bool]] %[[EQ_A1]] 12 %[[EQ_B1]] 7
  %7 = zext i1 %6 to i8
  store i8 %7, ptr %3, align 1, !tbaa !37
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_recipILi9ELi29ELi9ELi29EEvv() #3 {
  %1 = alloca i39, align 8
  %2 = alloca i39, align 8
  %3 = load i39, ptr %1, align 8, !tbaa !49
  %4 = call spir_func i39 @_Z32__spirv_ArbitraryFloatRecipINTELILi39ELi39EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i39 %3, i32 29, i32 29, i32 0, i32 2, i32 1) #5
; CHECK: %[[Recip_A1:[0-9]+]] = OpLoad %[[Ty_39]] %[[Recip_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[RecipResult:[0-9]+]] = OpArbitraryFloatRecipINTEL %[[Ty_39]] %[[Recip_A1]] 29 29 0 2 1
  store i39 %4, ptr %2, align 8, !tbaa !49
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_rsqrtILi12ELi19ELi13ELi20EEvv() #3 {
  %1 = alloca i32, align 4
  %2 = alloca i34, align 8
  %3 = load i32, ptr %1, align 4, !tbaa !51
  %4 = call spir_func i34 @_Z32__spirv_ArbitraryFloatRSqrtINTELILi32ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i32 %3, i32 19, i32 20, i32 0, i32 2, i32 1) #5
; CHECK: %[[Rsqrt_A1:[0-9]+]] = OpLoad %[[Ty_32]] %[[Rsqrt_AId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[RsqrtResult:[0-9]+]] = OpArbitraryFloatRSqrtINTEL %[[Ty_34]] %[[Rsqrt_A1]] 19 20 0 2 1
  store i34 %4, ptr %2, align 8, !tbaa !53
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_cbrtILi0ELi1ELi0ELi1EEvv() #3 {
  %1 = alloca i2, align 1
  %2 = alloca i2, align 1
  %3 = load i2, ptr %1, align 1, !tbaa !55
  %4 = call spir_func signext i2 @_Z31__spirv_ArbitraryFloatCbrtINTELILi2ELi2EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i2 signext %3, i32 1, i32 1, i32 0, i32 2, i32 1) #5
; CHECK: %[[Cbrt_A1:[0-9]+]] = OpLoad %[[Ty_2]] %[[Cbrt_AId:[0-9]+]] Aligned 1
; CHECK-NEXT: %[[CbrtResult:[0-9]+]] = OpArbitraryFloatCbrtINTEL %[[Ty_2]] %[[Cbrt_A1]] 1 1 0 2 1
  store i2 %4, ptr %2, align 1, !tbaa !55
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_hypotILi20ELi20ELi21ELi21ELi19ELi22EEvv() #3 {
  %1 = alloca i41, align 8
  %2 = alloca i43, align 8
  %3 = alloca i42, align 8
  %4 = load i41, ptr %1, align 8, !tbaa !57
  %5 = load i43, ptr %2, align 8, !tbaa !11
  %6 = call spir_func i42 @_Z32__spirv_ArbitraryFloatHypotINTELILi41ELi43ELi42EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i41 %4, i32 20, i43 %5, i32 21, i32 22, i32 0, i32 2, i32 1) #5
; CHECK: %[[Hypot_A1:[0-9]+]] = OpLoad %[[Ty_41]] %[[Hypot_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[Hypot_B1:[0-9]+]] = OpLoad %[[Ty_43]] %[[Hypot_BId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[HypotResult:[0-9]+]] = OpArbitraryFloatHypotINTEL %[[Ty_42]] %[[Hypot_A1]] 20 %[[Hypot_B1]] 21 22 0 2 1
  store i42 %6, ptr %3, align 8, !tbaa !59
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_sqrtILi7ELi7ELi8ELi8EEvv() #3 {
  %1 = alloca i15, align 2
  %2 = alloca i17, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr %2) #5
  %3 = load i15, ptr %1, align 2, !tbaa !21
  %4 = call spir_func signext i17 @_Z31__spirv_ArbitraryFloatSqrtINTELILi15ELi17EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i15 signext %3, i32 7, i32 8, i32 0, i32 2, i32 1) #5
; CHECK: %[[Sqrt_A1:[0-9]+]] = OpLoad %[[Ty_15]] %[[Sqrt_AId:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[SqrtResult:[0-9]+]] = OpArbitraryFloatSqrtINTEL %[[Ty_17]] %[[Sqrt_A1]] 7 8 0 2 1
  store i17 %4, ptr %2, align 4, !tbaa !61
  call void @llvm.lifetime.end.p0(i64 4, ptr %2) #5
  ret void
}

declare dso_local spir_func i40 @_Z31__spirv_ArbitraryFloatCastINTELILi40ELi40EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i40, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i25 @_Z38__spirv_ArbitraryFloatCastFromIntINTELILi43ELi25EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiibiii(i43, i32, i1 zeroext, i32, i32, i32) #4
declare dso_local spir_func signext i30 @_Z36__spirv_ArbitraryFloatCastToIntINTELILi23ELi30EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiibiii(i23 signext, i32, i1 zeroext, i32, i32, i32) #4
declare dso_local spir_func signext i14 @_Z30__spirv_ArbitraryFloatAddINTELILi13ELi15ELi14EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i13 signext, i32, i15 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i13 @_Z30__spirv_ArbitraryFloatAddINTELILi15ELi14ELi13EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i15 signext, i32, i14 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i13 @_Z30__spirv_ArbitraryFloatSubINTELILi9ELi11ELi13EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i9 signext, i32, i11 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i51 @_Z30__spirv_ArbitraryFloatMulINTELILi51ELi51ELi51EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i51, i32, i51, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i18 @_Z30__spirv_ArbitraryFloatDivINTELILi16ELi16ELi18EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i16 signext, i32, i16 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatGTINTELILi63ELi63EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i63, i32, i63, i32) #4
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatGEINTELILi47ELi47EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i47, i32, i47, i32) #4
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatLTINTELILi5ELi7EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i5 signext, i32, i7 signext, i32) #4
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatLEINTELILi55ELi55EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i55, i32, i55, i32) #4
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatEQINTELILi20ELi15EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i20 signext, i32, i15 signext, i32) #4
declare dso_local spir_func i39 @_Z32__spirv_ArbitraryFloatRecipINTELILi39ELi39EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i39, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i34 @_Z32__spirv_ArbitraryFloatRSqrtINTELILi32ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i32, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i2 @_Z31__spirv_ArbitraryFloatCbrtINTELILi2ELi2EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i2 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i42 @_Z32__spirv_ArbitraryFloatHypotINTELILi41ELi43ELi42EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i41, i32, i43, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i17 @_Z31__spirv_ArbitraryFloatSqrtINTELILi15ELi17EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i15 signext, i32, i32, i32, i32, i32) #4

!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"_ExtInt(40)", !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"_ExtInt(43)", !7, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"_ExtInt(25)", !7, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"_ExtInt(23)", !7, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"_ExtInt(30)", !7, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"_ExtInt(13)", !7, i64 0}
!21 = !{!22, !22, i64 0}
!22 = !{!"_ExtInt(15)", !7, i64 0}
!23 = !{!24, !24, i64 0}
!24 = !{!"_ExtInt(14)", !7, i64 0}
!25 = !{!26, !26, i64 0}
!26 = !{!"_ExtInt(9)", !7, i64 0}
!27 = !{!28, !28, i64 0}
!28 = !{!"_ExtInt(11)", !7, i64 0}
!29 = !{!30, !30, i64 0}
!30 = !{!"_ExtInt(51)", !7, i64 0}
!31 = !{!32, !32, i64 0}
!32 = !{!"_ExtInt(16)", !7, i64 0}
!33 = !{!34, !34, i64 0}
!34 = !{!"_ExtInt(18)", !7, i64 0}
!35 = !{!36, !36, i64 0}
!36 = !{!"_ExtInt(63)", !7, i64 0}
!37 = !{!38, !38, i64 0}
!38 = !{!"bool", !7, i64 0}
!39 = !{!40, !40, i64 0}
!40 = !{!"_ExtInt(47)", !7, i64 0}
!41 = !{!42, !42, i64 0}
!42 = !{!"_ExtInt(5)", !7, i64 0}
!43 = !{!44, !44, i64 0}
!44 = !{!"_ExtInt(7)", !7, i64 0}
!45 = !{!46, !46, i64 0}
!46 = !{!"_ExtInt(55)", !7, i64 0}
!47 = !{!48, !48, i64 0}
!48 = !{!"_ExtInt(20)", !7, i64 0}
!49 = !{!50, !50, i64 0}
!50 = !{!"_ExtInt(39)", !7, i64 0}
!51 = !{!52, !52, i64 0}
!52 = !{!"_ExtInt(32)", !7, i64 0}
!53 = !{!54, !54, i64 0}
!54 = !{!"_ExtInt(34)", !7, i64 0}
!55 = !{!56, !56, i64 0}
!56 = !{!"_ExtInt(2)", !7, i64 0}
!57 = !{!58, !58, i64 0}
!58 = !{!"_ExtInt(41)", !7, i64 0}
!59 = !{!60, !60, i64 0}
!60 = !{!"_ExtInt(42)", !7, i64 0}
!61 = !{!62, !62, i64 0}
!62 = !{!"_ExtInt(17)", !7, i64 0}
