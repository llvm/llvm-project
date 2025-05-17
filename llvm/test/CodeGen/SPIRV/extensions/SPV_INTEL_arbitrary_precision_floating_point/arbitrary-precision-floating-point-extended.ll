
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s 

; CHECK: OpCapability Kernel
; CHECK: OpCapability ArbitraryPrecisionFloatingPointINTEL
; CHECK: OpCapability ArbitraryPrecisionIntegersINTEL
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_floating_point"
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_integers"

; CHECK-DAG: %[[Ty_8:[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: %[[Ty_40:[0-9]+]] = OpTypeInt 40 0
; CHECK-DAG: %[[Ty_25:[0-9]+]] = OpTypeInt 25 0
; CHECK-DAG: %[[Ty_30:[0-9]+]] = OpTypeInt 30 0
; CHECK-DAG: %[[Ty_13:[0-9]+]] = OpTypeInt 13 0
; CHECK-DAG: %[[Ty_15:[0-9]+]] = OpTypeInt 15 0
; CHECK-DAG: %[[Ty_14:[0-9]+]] = OpTypeInt 14 0
; CHECK-DAG: %[[Ty_11:[0-9]+]] = OpTypeInt 11 0
; CHECK-DAG: %[[Ty_5:[0-9]+]] = OpTypeInt 5 0
; CHECK-DAG: %[[Ty_7:[0-9]+]] = OpTypeInt 7 0
; CHECK-DAG: %[[Ty_55:[0-9]+]] = OpTypeInt 55 0
; CHECK-DAG: %[[Ty_32:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[Ty_34:[0-9]+]] = OpTypeInt 34 0
; CHECK-DAG: %[[Ty_42:[0-9]+]] = OpTypeInt 42 0
; CHECK-DAG: %[[Ty_17:[0-9]+]] = OpTypeInt 17 0
; CHECK-DAG: %[[Ty_50:[0-9]+]] = OpTypeInt 50 0
; CHECK-DAG: %[[Ty_38:[0-9]+]] = OpTypeInt 38 0
; CHECK-DAG: %[[Ty_10:[0-9]+]] = OpTypeInt 10 0
; CHECK-DAG: %[[Ty_48:[0-9]+]] = OpTypeInt 48 0
; CHECK-DAG: %[[Ty_49:[0-9]+]] = OpTypeInt 49 0
; CHECK-DAG: %[[Ty_3:[0-9]+]] = OpTypeInt 3 0
; CHECK-DAG: %[[Ty_64:[0-9]+]] = OpTypeInt 64 0
; CHECK-DAG: %[[Ty_62:[0-9]+]] = OpTypeInt 62 0
; CHECK-DAG: %[[Ty_4:[0-9]+]] = OpTypeInt 4 0
; CHECK-DAG: %[[Ty_27:[0-9]+]] = OpTypeInt 27 0
; CHECK-DAG: %[[Ty_59:[0-9]+]] = OpTypeInt 59 0
; CHECK-DAG: %[[Ty_35:[0-9]+]] = OpTypeInt 35 0
; CHECK-DAG: %[[Ty_44:[0-9]+]] = OpTypeInt 44 0
; CHECK-DAG: %[[Ty_24:[0-9]+]] = OpTypeInt 24 0
; CHECK-DAG: %[[Ty_19:[0-9]+]] = OpTypeInt 19 0
; CHECK-DAG: %[[Ty_21:[0-9]+]] = OpTypeInt 21 0
; CHECK-DAG: %[[Ty_54:[0-9]+]] = OpTypeInt 54 0
; CHECK-DAG: %[[Ty_56:[0-9]+]] = OpTypeInt 56 0
; CHECK-DAG: %[[Ty_12:[0-9]+]] = OpTypeInt 12 0
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
  call spir_func void @_Z12ap_float_logILi30ELi19ELi19ELi30EEvv()
  call spir_func void @_Z13ap_float_log2ILi17ELi20ELi18ELi19EEvv()
  call spir_func void @_Z14ap_float_log10ILi4ELi3ELi4ELi5EEvv()
  call spir_func void @_Z14ap_float_log1pILi17ELi30ELi18ELi30EEvv()
  call spir_func void @_Z12ap_float_expILi16ELi25ELi16ELi25EEvv()
  call spir_func void @_Z13ap_float_exp2ILi1ELi1ELi2ELi2EEvv()
  call spir_func void @_Z14ap_float_exp10ILi8ELi16ELi8ELi16EEvv()
  call spir_func void @_Z14ap_float_expm1ILi21ELi42ELi20ELi41EEvv()
  call spir_func void @_Z12ap_float_sinILi14ELi15ELi16ELi17EEvv()
  call spir_func void @_Z12ap_float_cosILi1ELi2ELi2ELi1EEvv()
  call spir_func void @_Z15ap_float_sincosILi8ELi18ELi10ELi20EEvv()
  call spir_func void @_Z14ap_float_sinpiILi3ELi6ELi6ELi6EEvv()
  call spir_func void @_Z14ap_float_cospiILi18ELi40ELi18ELi40EEvv()
  call spir_func void @_Z17ap_float_sincospiILi9ELi20ELi11ELi20EEvv()
  call spir_func void @_Z13ap_float_asinILi2ELi4ELi2ELi8EEvv()
  call spir_func void @_Z15ap_float_asinpiILi11ELi23ELi11ELi23EEvv()
  call spir_func void @_Z13ap_float_acosILi4ELi9ELi3ELi10EEvv()
  call spir_func void @_Z15ap_float_acospiILi2ELi5ELi3ELi4EEvv()
  call spir_func void @_Z13ap_float_atanILi12ELi31ELi12ELi31EEvv()
  call spir_func void @_Z15ap_float_atanpiILi1ELi38ELi1ELi32EEvv()
  call spir_func void @_Z14ap_float_atan2ILi7ELi16ELi7ELi17ELi8ELi18EEvv()
  call spir_func void @_Z12ap_float_powILi8ELi8ELi9ELi9ELi10ELi10EEvv()
  call spir_func void @_Z13ap_float_powrILi18ELi35ELi19ELi35ELi20ELi35EEvv()
  call spir_func void @_Z13ap_float_pownILi4ELi7ELi10ELi5ELi9EEvv()
  call spir_func void @_Z15ap_float_sincosILi8ELi18ELi10ELi20EEvv_()
  call spir_func void @_Z14ap_float_atan2ILi7ELi16ELi7ELi17ELi8ELi18EEvv_()
  call spir_func void @_Z13ap_float_pownILi64ELi7ELi10ELi5ELi9EEvv()
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_logILi30ELi19ELi19ELi30EEvv() #3 {
  %1 = alloca i50, align 8
  %2 = alloca i50, align 8
  %3 = load i50, ptr %1, align 8, !tbaa !63
  %4 = call spir_func i50 @_Z30__spirv_ArbitraryFloatLogINTELILi50ELi50EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i50 %3, i32 19, i32 30, i32 0, i32 2, i32 1) #5
; CHECK: %[[Log_A1:[0-9]+]] = OpLoad %[[Ty_50]] %[[Log_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[LogResult:[0-9]+]] = OpArbitraryFloatLogINTEL %[[Ty_50]] %[[Log_A1]] 19 30 0 2 1
  store i50 %4, ptr %2, align 8, !tbaa !63
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_log2ILi17ELi20ELi18ELi19EEvv() #3 {
  %1 = alloca i38, align 8
  %2 = alloca i38, align 8
  %3 = load i38, ptr %1, align 8, !tbaa !65
  %4 = call spir_func i38 @_Z31__spirv_ArbitraryFloatLog2INTELILi38ELi38EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i38 %3, i32 20, i32 19, i32 0, i32 2, i32 1) #5
; CHECK: %[[Log2_A1:[0-9]+]] = OpLoad %[[Ty_38]] %[[Log2_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[Log2Result:[0-9]+]] = OpArbitraryFloatLog2INTEL %[[Ty_38]] %[[Log2_A1]] 20 19 0 2 1
  store i38 %4, ptr %2, align 8, !tbaa !65
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_log10ILi4ELi3ELi4ELi5EEvv() #3 {
  %1 = alloca i8, align 1
  %2 = alloca i10, align 2
  call void @llvm.lifetime.start.p0(i64 1, ptr %1) #5
  %3 = load i8, ptr %1, align 1, !tbaa !67
  %4 = call spir_func signext i10 @_Z32__spirv_ArbitraryFloatLog10INTELILi8ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i8 signext %3, i32 3, i32 5, i32 0, i32 2, i32 1) #5
; CHECK: %[[Log10_A1:[0-9]+]] = OpLoad %[[Ty_8]] %[[Log10_AId:[0-9]+]] Aligned 1
; CHECK-NEXT: %[[Log10Result:[0-9]+]] = OpArbitraryFloatLog10INTEL %[[Ty_10]] %[[Log10_A1]] 3 5 0 2 1
  store i10 %4, ptr %2, align 2, !tbaa !69
  call void @llvm.lifetime.end.p0(i64 1, ptr %1) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_log1pILi17ELi30ELi18ELi30EEvv() #3 {
  %1 = alloca i48, align 8
  %2 = alloca i49, align 8
  %3 = load i48, ptr %1, align 8, !tbaa !71
  %4 = call spir_func i49 @_Z32__spirv_ArbitraryFloatLog1pINTELILi48ELi49EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i48 %3, i32 30, i32 30, i32 0, i32 2, i32 1) #5
; CHECK: %[[Log1p_A1:[0-9]+]] = OpLoad %[[Ty_48]] %[[Log1p_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[Log1pResult:[0-9]+]] = OpArbitraryFloatLog1pINTEL %[[Ty_49]] %[[Log1p_A1]] 30 30 0 2 1
  store i49 %4, ptr %2, align 8, !tbaa !73
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_expILi16ELi25ELi16ELi25EEvv() #3 {
  %1 = alloca i42, align 8
  %2 = alloca i42, align 8
  %3 = load i42, ptr %1, align 8, !tbaa !59
  %4 = call spir_func i42 @_Z30__spirv_ArbitraryFloatExpINTELILi42ELi42EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i42 %3, i32 25, i32 25, i32 0, i32 2, i32 1) #5
; CHECK: %[[Exp_A1:[0-9]+]] = OpLoad %[[Ty_42]] %[[Exp_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[ExpResult:[0-9]+]] = OpArbitraryFloatExpINTEL %[[Ty_42]] %[[Exp_A1]] 25 25 0 2 1
  store i42 %4, ptr %2, align 8, !tbaa !59
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_exp2ILi1ELi1ELi2ELi2EEvv() #3 {
  %1 = alloca i3, align 1
  %2 = alloca i5, align 1
  %3 = load i3, ptr %1, align 1, !tbaa !75
  %4 = call spir_func signext i5 @_Z31__spirv_ArbitraryFloatExp2INTELILi3ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i3 signext %3, i32 1, i32 2, i32 0, i32 2, i32 1) #5
; CHECK: %[[Exp2_A1:[0-9]+]] = OpLoad %[[Ty_3]] %[[Exp2_AId:[0-9]+]] Aligned 1
; CHECK-NEXT: %[[Exp2Result:[0-9]+]] = OpArbitraryFloatExp2INTEL %[[Ty_5]] %[[Exp2_A1]] 1 2 0 2 1
  store i5 %4, ptr %2, align 1, !tbaa !41
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_exp10ILi8ELi16ELi8ELi16EEvv() #3 {
  %1 = alloca i25, align 4
  %2 = alloca i25, align 4
  %3 = load i25, ptr %1, align 4, !tbaa !13
  %4 = call spir_func signext i25 @_Z32__spirv_ArbitraryFloatExp10INTELILi25ELi25EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i25 signext %3, i32 16, i32 16, i32 0, i32 2, i32 1) #5
; CHECK: %[[Exp10_A1:[0-9]+]] = OpLoad %[[Ty_25]] %[[Exp10_AId:[0-9]+]] Aligned 
; CHECK-NEXT: %[[Exp10Result:[0-9]+]] = OpArbitraryFloatExp10INTEL %[[Ty_25]] %[[Exp10_A1]] 16 16 0 2 1
  store i25 %4, ptr %2, align 4, !tbaa !13
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_expm1ILi21ELi42ELi20ELi41EEvv() #3 {
  %1 = alloca i64, align 8
  %2 = alloca i62, align 8
  %3 = load i64, ptr %1, align 8, !tbaa !77
  %4 = call spir_func i62 @_Z32__spirv_ArbitraryFloatExpm1INTELILi64ELi62EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i64 %3, i32 42, i32 41, i32 0, i32 2, i32 1) #5
; CHECK: %[[Expm1_A1:[0-9]+]] = OpLoad %[[Ty_64]] %[[Expm1_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[Expm1Result:[0-9]+]] = OpArbitraryFloatExpm1INTEL %[[Ty_62]] %[[Expm1_A1]] 42 41 0 2 1
  store i62 %4, ptr %2, align 8, !tbaa !79
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_sinILi14ELi15ELi16ELi17EEvv() #3 {
  %1 = alloca i30, align 4
  %2 = alloca i34, align 8
  %3 = load i30, ptr %1, align 4, !tbaa !17
  %4 = call spir_func i34 @_Z30__spirv_ArbitraryFloatSinINTELILi30ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i30 signext %3, i32 15, i32 17, i32 0, i32 2, i32 1) #5
; CHECK: %[[Sin_A1:[0-9]+]] = OpLoad %[[Ty_30]] %[[Sin_AId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[SinResult:[0-9]+]] = OpArbitraryFloatSinINTEL %[[Ty_34]] %[[Sin_A1]] 15 17 0 2 1
  store i34 %4, ptr %2, align 8, !tbaa !53
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_cosILi1ELi2ELi2ELi1EEvv() #3 {
  %1 = alloca i4, align 1
  %2 = alloca i4, align 1
  %3 = load i4, ptr %1, align 1, !tbaa !81
  %4 = call spir_func signext i4 @_Z30__spirv_ArbitraryFloatCosINTELILi4ELi4EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i4 signext %3, i32 2, i32 1, i32 0, i32 2, i32 1) #5
; CHECK: %[[Cos_A1:[0-9]+]] = OpLoad %[[Ty_4]] %[[Cos_AId:[0-9]+]] Aligned 1
; CHECK-NEXT: %[[CosResult:[0-9]+]] = OpArbitraryFloatCosINTEL %[[Ty_4]] %[[Cos_A1]] 2 1 0 2 1
  store i4 %4, ptr %2, align 1, !tbaa !81
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z15ap_float_sincosILi8ELi18ELi10ELi20EEvv() #3 {
  %1 = alloca i27, align 4
  %2 = alloca i62, align 8
  %3 = load i27, ptr %1, align 4, !tbaa !83
  %4 = call spir_func i62 @_Z33__spirv_ArbitraryFloatSinCosINTELILi27ELi31EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(i27 signext %3, i32 18, i32 20, i32 0, i32 2, i32 1) #5
; CHECK: %[[SinCos_A1:[0-9]+]] = OpLoad %[[Ty_27]] %[[SinCos_AId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[SinCosResult:[0-9]+]] = OpArbitraryFloatSinCosINTEL %[[Ty_62]] %[[SinCos_A1]] 18 20 0 2 1
  store i62 %4, ptr %2, align 8, !tbaa !79
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_sinpiILi3ELi6ELi6ELi6EEvv() #3 {
  %1 = alloca i10, align 2
  %2 = alloca i13, align 2
  %3 = load i10, ptr %1, align 2, !tbaa !69
  %4 = call spir_func signext i13 @_Z32__spirv_ArbitraryFloatSinPiINTELILi10ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i10 signext %3, i32 6, i32 6, i32 0, i32 2, i32 1) #5
; CHECK: %[[SinPi_A1:[0-9]+]] = OpLoad %[[Ty_10]] %[[SinPi_AId:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[SinPiResult:[0-9]+]] = OpArbitraryFloatSinPiINTEL %[[Ty_13]] %[[SinPi_A1]] 6 6 0 2 1
  store i13 %4, ptr %2, align 2, !tbaa !19
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_cospiILi18ELi40ELi18ELi40EEvv() #3 {
  %1 = alloca i59, align 8
  %2 = alloca i59, align 8
  %3 = load i59, ptr %1, align 8, !tbaa !85
  %4 = call spir_func i59 @_Z32__spirv_ArbitraryFloatCosPiINTELILi59ELi59EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i59 %3, i32 40, i32 40, i32 0, i32 2, i32 1) #5
; CHECK: %[[CosPi_A1:[0-9]+]] = OpLoad %[[Ty_59]] %[[CosPi_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[CosPiResult:[0-9]+]] = OpArbitraryFloatCosPiINTEL %[[Ty_59]] %[[CosPi_A1]] 40 40 0 2 1
  store i59 %4, ptr %2, align 8, !tbaa !85
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z17ap_float_sincospiILi9ELi20ELi11ELi20EEvv() #3 {
  %1 = alloca i30, align 4
  %2 = alloca i64, align 8
  %3 = load i30, ptr %1, align 4, !tbaa !17
  %4 = call spir_func i64 @_Z35__spirv_ArbitraryFloatSinCosPiINTELILi30ELi32EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(i30 signext %3, i32 20, i32 20, i32 0, i32 2, i32 1) #5
; CHECK: %[[SinCosPi_A1:[0-9]+]] = OpLoad %[[Ty_30]] %[[SinCosPi_AId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[SinCosPiResult:[0-9]+]] = OpArbitraryFloatSinCosPiINTEL %[[Ty_64]] %[[SinCosPi_A1]] 20 20 0 2 1
  store i64 %4, ptr %2, align 8, !tbaa !77
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_asinILi2ELi4ELi2ELi8EEvv() #3 {
  %1 = alloca i7, align 1
  %2 = alloca i11, align 2
  call void @llvm.lifetime.start.p0(i64 1, ptr %1) #5
  %3 = load i7, ptr %1, align 1, !tbaa !43
  %4 = call spir_func signext i11 @_Z31__spirv_ArbitraryFloatASinINTELILi7ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i7 signext %3, i32 4, i32 8, i32 0, i32 2, i32 1) #5
; CHECK: %[[ASin_A1:[0-9]+]] = OpLoad %[[Ty_7]] %[[ASin_AId:[0-9]+]] Aligned 1
; CHECK-NEXT: %[[ASinResult:[0-9]+]] = OpArbitraryFloatASinINTEL %[[Ty_11]] %[[ASin_A1]] 4 8 0 2 1
  store i11 %4, ptr %2, align 2, !tbaa !27
  call void @llvm.lifetime.end.p0(i64 1, ptr %1) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z15ap_float_asinpiILi11ELi23ELi11ELi23EEvv() #3 {
  %1 = alloca i35, align 8
  %2 = alloca i35, align 8
  %3 = load i35, ptr %1, align 8, !tbaa !87
  %4 = call spir_func i35 @_Z33__spirv_ArbitraryFloatASinPiINTELILi35ELi35EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i35 %3, i32 23, i32 23, i32 0, i32 2, i32 1) #5
; CHECK: %[[ASinPi_A1:[0-9]+]] = OpLoad %[[Ty_35]] %[[ASinPi_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[ASinPiResult:[0-9]+]] = OpArbitraryFloatASinPiINTEL %[[Ty_35]] %[[ASinPi_A1]] 23 23 0 2 1
  store i35 %4, ptr %2, align 8, !tbaa !87
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_acosILi4ELi9ELi3ELi10EEvv() #3 {
  %1 = alloca i14, align 2
  %2 = alloca i14, align 2
  %3 = load i14, ptr %1, align 2, !tbaa !23
  %4 = call spir_func signext i14 @_Z31__spirv_ArbitraryFloatACosINTELILi14ELi14EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i14 signext %3, i32 9, i32 10, i32 0, i32 2, i32 1) #5
; CHECK: %[[ACos_A1:[0-9]+]] = OpLoad %[[Ty_14]] %[[ACos_AId:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[ACosResult:[0-9]+]] = OpArbitraryFloatACosINTEL %[[Ty_14]] %[[ACos_A1]] 9 10 0 2 1
  store i14 %4, ptr %2, align 2, !tbaa !23
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z15ap_float_acospiILi2ELi5ELi3ELi4EEvv() #3 {
  %1 = alloca i8, align 1
  %2 = alloca i8, align 1
  %3 = load i8, ptr %1, align 1, !tbaa !67
  %4 = call spir_func signext i8 @_Z33__spirv_ArbitraryFloatACosPiINTELILi8ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i8 signext %3, i32 5, i32 4, i32 0, i32 2, i32 1) #5
; CHECK: %[[ACosPi_A1:[0-9]+]] = OpLoad %[[Ty_8]] %[[ACosPi_AId:[0-9]+]] Aligned 1
; CHECK-NEXT: %[[ACosPiResult:[0-9]+]] = OpArbitraryFloatACosPiINTEL %[[Ty_8]] %[[ACosPi_A1]] 5 4 0 2 1
  store i8 %4, ptr %2, align 1, !tbaa !67
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_atanILi12ELi31ELi12ELi31EEvv() #3 {
  %1 = alloca i44, align 8
  %2 = alloca i44, align 8
  %3 = load i44, ptr %1, align 8, !tbaa !89
  %4 = call spir_func i44 @_Z31__spirv_ArbitraryFloatATanINTELILi44ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i44 %3, i32 31, i32 31, i32 0, i32 2, i32 1) #5
; CHECK: %[[ATan_A1:[0-9]+]] = OpLoad %[[Ty_44]] %[[ATan_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[ATanResult:[0-9]+]] = OpArbitraryFloatATanINTEL %[[Ty_44]] %[[ATan_A1]] 31 31 0 2 1
  store i44 %4, ptr %2, align 8, !tbaa !89
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z15ap_float_atanpiILi1ELi38ELi1ELi32EEvv() #3 {
  %1 = alloca i40, align 8
  %2 = alloca i34, align 8
  %3 = load i40, ptr %1, align 8, !tbaa !9
  %4 = call spir_func i34 @_Z33__spirv_ArbitraryFloatATanPiINTELILi40ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i40 %3, i32 38, i32 32, i32 0, i32 2, i32 1) #5
; CHECK: %[[ATanPi_A1:[0-9]+]] = OpLoad %[[Ty_40]] %[[ATanPi_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[ATanPiResult:[0-9]+]] = OpArbitraryFloatATanPiINTEL %[[Ty_34]] %[[ATanPi_A1]] 38 32 0 2 1
  store i34 %4, ptr %2, align 8, !tbaa !53
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_atan2ILi7ELi16ELi7ELi17ELi8ELi18EEvv() #3 {
  %1 = alloca i24, align 4
  %2 = alloca i25, align 4
  %3 = alloca i27, align 4
  %4 = load i24, ptr %1, align 4, !tbaa !91
  %5 = load i25, ptr %2, align 4, !tbaa !13
  %6 = call spir_func signext i27 @_Z32__spirv_ArbitraryFloatATan2INTELILi24ELi25ELi27EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i24 signext %4, i32 16, i25 signext %5, i32 17, i32 18, i32 0, i32 2, i32 1) #5
; CHECK: %[[ATan2_A1:[0-9]+]] = OpLoad %[[Ty_24]] %[[ATan2_AId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[ATan2_B1:[0-9]+]] = OpLoad %[[Ty_25]] %[[ATan2_BId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[ATan2Result:[0-9]+]] = OpArbitraryFloatATan2INTEL %[[Ty_27]] %[[ATan2_A1]] 16 %[[ATan2_B1]] 17 18 0 2 1
  store i27 %6, ptr %3, align 4, !tbaa !83
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z12ap_float_powILi8ELi8ELi9ELi9ELi10ELi10EEvv() #3 {
  %1 = alloca i17, align 4
  %2 = alloca i19, align 4
  %3 = alloca i21, align 4
  %4 = load i17, ptr %1, align 4, !tbaa !61
  %5 = load i19, ptr %2, align 4, !tbaa !93
  %6 = call spir_func signext i21 @_Z30__spirv_ArbitraryFloatPowINTELILi17ELi19ELi21EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i17 signext %4, i32 8, i19 signext %5, i32 9, i32 10, i32 0, i32 2, i32 1) #5
; CHECK: %[[Pow_A1:[0-9]+]] = OpLoad %[[Ty_17]] %[[Pow_AId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[Pow_B1:[0-9]+]] = OpLoad %[[Ty_19]] %[[Pow_BId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[PowResult:[0-9]+]] = OpArbitraryFloatPowINTEL %[[Ty_21]] %[[Pow_A1]] 8 %[[Pow_B1]] 9 10 0 2 1
  store i21 %6, ptr %3, align 4, !tbaa !95
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_powrILi18ELi35ELi19ELi35ELi20ELi35EEvv() #3 {
  %1 = alloca i54, align 8
  %2 = alloca i55, align 8
  %3 = alloca i56, align 8
  %4 = load i54, ptr %1, align 8, !tbaa !97
  %5 = load i55, ptr %2, align 8, !tbaa !45
  %6 = call spir_func i56 @_Z31__spirv_ArbitraryFloatPowRINTELILi54ELi55ELi56EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i54 %4, i32 35, i55 %5, i32 35, i32 35, i32 0, i32 2, i32 1) #5
; CHECK: %[[PowR_A1:[0-9]+]] = OpLoad %[[Ty_54]] %[[PowR_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[PowR_B1:[0-9]+]] = OpLoad %[[Ty_55]] %[[PowR_BId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[PowRResult:[0-9]+]] = OpArbitraryFloatPowRINTEL %[[Ty_56]] %[[PowR_A1]] 35 %[[PowR_B1]] 35 35 0 2 1
  store i56 %6, ptr %3, align 8, !tbaa !99
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_pownILi4ELi7ELi10ELi5ELi9EEvv() #3 {
  %1 = alloca i12, align 2
  %2 = alloca i10, align 2
  %3 = alloca i15, align 2
  %4 = load i12, ptr %1, align 2, !tbaa !101
  %5 = load i10, ptr %2, align 2, !tbaa !69
  %6 = call spir_func signext i15 @_Z31__spirv_ArbitraryFloatPowNINTELILi12ELi10ELi15EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiii(i12 signext %4, i32 7, i10 signext %5, i1 zeroext false, i32 9, i32 0, i32 2, i32 1) #5
; CHECK: %[[PowN_A1:[0-9]+]] = OpLoad %[[Ty_12]] %[[PowN_AId:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[PowN_B1:[0-9]+]] = OpLoad %[[Ty_10]] %[[PowN_BId:[0-9]+]] Aligned 2
; CHECK-NEXT: %[[PowNResult:[0-9]+]] = OpArbitraryFloatPowNINTEL %[[Ty_15]] %[[PowN_A1]] 7 %[[PowN_B1]] 0 9 0 2 1
  store i15 %6, ptr %3, align 2, !tbaa !21
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z15ap_float_sincosILi8ELi18ELi10ELi20EEvv_() #3 {
  %1 = alloca i34, align 8
  %2 = addrspacecast ptr %1 to ptr addrspace(4)
  %3 = alloca i64, align 8
  %4 = addrspacecast ptr %3 to ptr addrspace(4)
  %5 = load i34, ptr addrspace(4) %2, align 8
  call spir_func void @_Z33__spirv_ArbitraryFloatSinCosINTELILi34ELi64EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(ptr addrspace(4) sret(i64) align 8 %4, i34 %5, i32 18, i32 20, i32 0, i32 2, i32 1) #5
; CHECK: %[[SinCos_A1:[0-9]+]] = OpLoad %[[Ty_34]] %[[SinCos_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[SinCosResult:[0-9]+]] = OpArbitraryFloatSinCosINTEL %[[Ty_64]] %[[SinCos_A1]] 18 20 0 2 1
; CHECK-NEXT: OpStore %[[#]] %[[SinCosResult]] Aligned 4
  %6 = load i64, ptr addrspace(4) %4, align 8
  store i64 %6, ptr addrspace(4) %4, align 8
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z14ap_float_atan2ILi7ELi16ELi7ELi17ELi8ELi18EEvv_() #3 {
  %1 = alloca i24, align 4
  %2 = alloca i25, align 4
  %3 = alloca i64, align 8
  %4 = addrspacecast ptr %3 to ptr addrspace(4)
  %5 = load i24, ptr %1, align 4, !tbaa !91
  %6 = load i25, ptr %2, align 4, !tbaa !13
  call spir_func void @_Z32__spirv_ArbitraryFloatATan2INTELILi24ELi25ELi64EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(ptr addrspace(4) sret(i64) align 8 %4, i24 signext %5, i32 16, i25 signext %6, i32 17, i32 18, i32 0, i32 2, i32 1) #5
; CHECK: %[[ATan2_A1:[0-9]+]] = OpLoad %[[Ty_24]] %[[ATan2_AId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[ATan2_B1:[0-9]+]] = OpLoad %[[Ty_25]] %[[ATan2_BId:[0-9]+]] Aligned 4
; CHECK-NEXT: %[[ATan2Result:[0-9]+]] = OpArbitraryFloatATan2INTEL %[[Ty_64]] %[[ATan2_A1]] 16 %[[ATan2_B1]] 17 18 0 2 1
; CHECK-NEXT: OpStore %[[#]] %[[ATan2Result]]
  %7 = load i64, ptr addrspace(4) %4, align 8
  store i64 %7, ptr addrspace(4) %4, align 8
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z13ap_float_pownILi64ELi7ELi10ELi5ELi9EEvv() #3 {
entry:
  %A = alloca i64, align 8
  %A.ascast = addrspacecast ptr %A to ptr addrspace(4)
  %B = alloca i10, align 2
  %B.ascast = addrspacecast ptr %B to ptr addrspace(4)
  %pown_res = alloca i15, align 2
  %pown_res.ascast = addrspacecast ptr %pown_res to ptr addrspace(4)
  %indirect-arg-temp = alloca i64, align 8
  %0 = load i64, ptr addrspace(4) %A.ascast, align 8
  %1 = load i10, ptr addrspace(4) %B.ascast, align 2
  store i64 %0, ptr %indirect-arg-temp, align 8
; CHECK: %[[#]] = OpLoad %[[Ty_64]] %[[#]] Aligned 8
; CHECK: %[[PowN_A1:[0-9]+]] = OpLoad %[[Ty_64]] %[[PowN_AId:[0-9]+]] Aligned 8
; CHECK-NEXT: %[[PowN_B1:[0-9]+]] = OpLoad %[[Ty_10]] %[[PowN_BId:[0-9]+]] Aligned 2
; CHECK-NEXT: OpStore %[[PtrId:[0-9]+]] %[[PowN_A1]] Aligned 8
; CHECK-NEXT: %[[PowNResult:[0-9]+]] = OpArbitraryFloatPowNINTEL %[[Ty_15]] %[[#]] 7 %[[PowN_B1]] 1 9 0 2 1
; CHECK-NEXT: OpStore %[[PtrId:[0-9]+]] %[[PowNResult]] Aligned 2
  %call = call spir_func signext i15 @_Z31__spirv_ArbitraryFloatPowNINTELILi64ELi10ELi15EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiii(ptr byval(i64) align 8 %indirect-arg-temp, i32 7, i10 signext %1, i1 zeroext true, i32 9, i32 0, i32 2, i32 1) #4
  store i15 %call, ptr addrspace(4) %pown_res.ascast, align 2
  ret void
}

declare dso_local spir_func i50 @_Z30__spirv_ArbitraryFloatLogINTELILi50ELi50EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i50, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i38 @_Z31__spirv_ArbitraryFloatLog2INTELILi38ELi38EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i38, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i10 @_Z32__spirv_ArbitraryFloatLog10INTELILi8ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i8 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i49 @_Z32__spirv_ArbitraryFloatLog1pINTELILi48ELi49EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i48, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i42 @_Z30__spirv_ArbitraryFloatExpINTELILi42ELi42EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i42, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i5 @_Z31__spirv_ArbitraryFloatExp2INTELILi3ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i3 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i25 @_Z32__spirv_ArbitraryFloatExp10INTELILi25ELi25EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i25 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i62 @_Z32__spirv_ArbitraryFloatExpm1INTELILi64ELi62EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i64, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i34 @_Z30__spirv_ArbitraryFloatSinINTELILi30ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i30 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i4 @_Z30__spirv_ArbitraryFloatCosINTELILi4ELi4EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i4 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i62 @_Z33__spirv_ArbitraryFloatSinCosINTELILi27ELi31EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(i27 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i13 @_Z32__spirv_ArbitraryFloatSinPiINTELILi10ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i10 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i59 @_Z32__spirv_ArbitraryFloatCosPiINTELILi59ELi59EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i59, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i64 @_Z35__spirv_ArbitraryFloatSinCosPiINTELILi30ELi32EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(i30 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i11 @_Z31__spirv_ArbitraryFloatASinINTELILi7ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i7 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i35 @_Z33__spirv_ArbitraryFloatASinPiINTELILi35ELi35EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i35, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i14 @_Z31__spirv_ArbitraryFloatACosINTELILi14ELi14EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i14 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i8 @_Z33__spirv_ArbitraryFloatACosPiINTELILi8ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i8 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i44 @_Z31__spirv_ArbitraryFloatATanINTELILi44ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i44, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i34 @_Z33__spirv_ArbitraryFloatATanPiINTELILi40ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i40, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i27 @_Z32__spirv_ArbitraryFloatATan2INTELILi24ELi25ELi27EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i24 signext, i32, i25 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i21 @_Z30__spirv_ArbitraryFloatPowINTELILi17ELi19ELi21EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i17 signext, i32, i19 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func i56 @_Z31__spirv_ArbitraryFloatPowRINTELILi54ELi55ELi56EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i54, i32, i55, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i15 @_Z31__spirv_ArbitraryFloatPowNINTELILi12ELi10ELi15EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiii(i12 signext, i32, i10 signext, i1 zeroext, i32, i32, i32, i32) #4
declare dso_local spir_func void @_Z33__spirv_ArbitraryFloatSinCosINTELILi34ELi64EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(ptr addrspace(4) sret(i64) align 8, i34, i32, i32, i32, i32, i32) #4
declare dso_local spir_func void @_Z32__spirv_ArbitraryFloatATan2INTELILi24ELi25ELi64EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(ptr addrspace(4) sret(i64) align 8, i24 signext, i32, i25 signext, i32, i32, i32, i32, i32) #4
declare dso_local spir_func signext i15 @_Z31__spirv_ArbitraryFloatPowNINTELILi64ELi10ELi15EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiii(ptr byval(i64) align 8, i32, i10 signext, i1 zeroext, i32, i32, i32, i32) #4

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
!63 = !{!64, !64, i64 0}
!64 = !{!"_ExtInt(50)", !7, i64 0}
!65 = !{!66, !66, i64 0}
!66 = !{!"_ExtInt(38)", !7, i64 0}
!67 = !{!68, !68, i64 0}
!68 = !{!"_ExtInt(8)", !7, i64 0}
!69 = !{!70, !70, i64 0}
!70 = !{!"_ExtInt(10)", !7, i64 0}
!71 = !{!72, !72, i64 0}
!72 = !{!"_ExtInt(48)", !7, i64 0}
!73 = !{!74, !74, i64 0}
!74 = !{!"_ExtInt(49)", !7, i64 0}
!75 = !{!76, !76, i64 0}
!76 = !{!"_ExtInt(3)", !7, i64 0}
!77 = !{!78, !78, i64 0}
!78 = !{!"_ExtInt(64)", !7, i64 0}
!79 = !{!80, !80, i64 0}
!80 = !{!"_ExtInt(62)", !7, i64 0}
!81 = !{!82, !82, i64 0}
!82 = !{!"_ExtInt(4)", !7, i64 0}
!83 = !{!84, !84, i64 0}
!84 = !{!"_ExtInt(27)", !7, i64 0}
!85 = !{!86, !86, i64 0}
!86 = !{!"_ExtInt(59)", !7, i64 0}
!87 = !{!88, !88, i64 0}
!88 = !{!"_ExtInt(35)", !7, i64 0}
!89 = !{!90, !90, i64 0}
!90 = !{!"_ExtInt(44)", !7, i64 0}
!91 = !{!92, !92, i64 0}
!92 = !{!"_ExtInt(24)", !7, i64 0}
!93 = !{!94, !94, i64 0}
!94 = !{!"_ExtInt(19)", !7, i64 0}
!95 = !{!96, !96, i64 0}
!96 = !{!"_ExtInt(21)", !7, i64 0}
!97 = !{!98, !98, i64 0}
!98 = !{!"_ExtInt(54)", !7, i64 0}
!99 = !{!100, !100, i64 0}
!100 = !{!"_ExtInt(56)", !7, i64 0}
!101 = !{!102, !102, i64 0}
!102 = !{!"_ExtInt(12)", !7, i64 0}
