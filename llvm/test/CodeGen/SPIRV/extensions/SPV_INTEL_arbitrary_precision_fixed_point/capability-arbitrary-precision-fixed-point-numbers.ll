; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_fixed_point,+SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s 
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_fixed_point,+SPV_INTEL_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}


; CHECK-DAG: OpCapability Kernel
; CHECK-DAG: OpCapability ArbitraryPrecisionIntegersINTEL
; CHECK-DAG: OpCapability ArbitraryPrecisionFixedPointINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_arbitrary_precision_fixed_point"
; CHECK-DAG: OpExtension "SPV_INTEL_arbitrary_precision_integers"

; CHECK-DAG: %[[Ty_8:[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: %[[Ty_13:[0-9]+]] = OpTypeInt 13 0
; CHECK-DAG: %[[Ty_5:[0-9]+]] = OpTypeInt 5 0
; CHECK-DAG: %[[Ty_3:[0-9]+]] = OpTypeInt 3 0
; CHECK-DAG: %[[Ty_11:[0-9]+]] = OpTypeInt 11 0
; CHECK-DAG: %[[Ty_10:[0-9]+]] = OpTypeInt 10 0
; CHECK-DAG: %[[Ty_17:[0-9]+]] = OpTypeInt 17 0
; CHECK-DAG: %[[Ty_35:[0-9]+]] = OpTypeInt 35 0
; CHECK-DAG: %[[Ty_28:[0-9]+]] = OpTypeInt 28 0
; CHECK-DAG: %[[Ty_31:[0-9]+]] = OpTypeInt 31 0
; CHECK-DAG: %[[Ty_40:[0-9]+]] = OpTypeInt 40 0
; CHECK-DAG: %[[Ty_60:[0-9]+]] = OpTypeInt 60 0
; CHECK-DAG: %[[Ty_16:[0-9]+]] = OpTypeInt 16 0
; CHECK-DAG: %[[Ty_64:[0-9]+]] = OpTypeInt 64 0
; CHECK-DAG: %[[Ty_44:[0-9]+]] = OpTypeInt 44 0
; CHECK-DAG: %[[Ty_34:[0-9]+]] = OpTypeInt 34 0
; CHECK-DAG: %[[Ty_51:[0-9]+]] = OpTypeInt 51 0



; CHECK:        %[[Sqrt_InId:[0-9]+]] = OpLoad %[[Ty_13]]
; CHECK-NEXT:  %[[#]] = OpFixedSqrtINTEL %[[Ty_5]] %[[Sqrt_InId]] 0 2 2 0 0

; CHECK:        %[[Sqrt_InId_B:[0-9]+]] = OpLoad %[[Ty_5]]
; CHECK-NEXT:  %[[#]] = OpFixedSqrtINTEL %[[Ty_13]] %[[Sqrt_InId_B]] 0 2 2 0 0

; CHECK:        %[[Sqrt_InId_C:[0-9]+]] = OpLoad %[[Ty_5]]
; CHECK-NEXT:  %[[#]] = OpFixedSqrtINTEL %[[Ty_13]] %[[Sqrt_InId_C]] 0 2 2 0 0


; CHECK:        %[[Recip_InId:[0-9]+]] = OpLoad %[[Ty_3]]
; CHECK-NEXT:  %[[#]] = OpFixedRecipINTEL %[[Ty_8]] %[[Recip_InId]] 1 4 4 0 0

; CHECK:        %[[Rsqrt_InId:[0-9]+]] = OpLoad %[[Ty_11]]
; CHECK-NEXT:  %[[#]] = OpFixedRsqrtINTEL %[[Ty_10]] %[[Rsqrt_InId]] 0 8 6 0 0

; CHECK:        %[[Sin_InId:[0-9]+]] = OpLoad %[[Ty_17]]
; CHECK-NEXT:  %[[#]] = OpFixedSinINTEL %[[Ty_11]] %[[Sin_InId]] 1 7 5 0 0

; CHECK:        %[[Cos_InId:[0-9]+]] = OpLoad %[[Ty_35]]
; CHECK-NEXT:  %[[#]] = OpFixedCosINTEL %[[Ty_28]] %[[Cos_InId]] 0 9 3 0 0

; CHECK:        %[[SinCos_InId:[0-9]+]] = OpLoad %[[Ty_31]]
; CHECK-NEXT:  %[[#]] = OpFixedSinCosINTEL %[[Ty_40]] %[[SinCos_InId]] 1 10 12 0 0

; CHECK:        %[[SinPi_InId:[0-9]+]] = OpLoad %[[Ty_60]]
; CHECK-NEXT:  %[[#]] = OpFixedSinPiINTEL %[[Ty_5]] %[[SinPi_InId]] 0 2 2 0 0

; CHECK:        %[[CosPi_InId:[0-9]+]] = OpLoad %[[Ty_28]]
; CHECK-NEXT:  %[[#]] = OpFixedCosPiINTEL %[[Ty_16]] %[[CosPi_InId]] 0 8 5 0 0

; CHECK:        %[[SinCosPi_InId:[0-9]+]] = OpLoad %[[Ty_13]]
; CHECK-NEXT:  %[[#]] = OpFixedSinCosPiINTEL %[[Ty_10]] %[[SinCosPi_InId]] 0 2 2 0 0

; CHECK:        %[[Log_InId:[0-9]+]] = OpLoad %[[Ty_64]]
; CHECK-NEXT:  %[[#]] = OpFixedLogINTEL %[[Ty_44]] %[[Log_InId]] 1 24 22 0 0

; CHECK:        %[[Exp_InId:[0-9]+]] = OpLoad %[[Ty_44]]
; CHECK-NEXT:  %[[#]] = OpFixedExpINTEL %[[Ty_34]] %[[Exp_InId]] 0 20 20 0 0


; CHECK:        %[[SinCos_InId:[0-9]+]] = OpLoad %[[Ty_34]]
; CHECK-NEXT:  %[[SinCos_ResultId:[0-9]+]] = OpFixedSinCosINTEL %[[Ty_51]] %[[SinCos_InId]] 1 3 2 0 0
; CHECK-NEXT:        OpStore %[[#]] %[[SinCos_ResultId]]

; CHECK:        %[[#]] = OpLabel 
; CHECK:        %[[ResId:[0-9]+]] = OpLoad %[[Ty_51]]
; CHECK-NEXT:  OpStore %[[PtrId:[0-9]+]] %[[ResId]]
; CHECK-NEXT:  %[[ExpInId2:[0-9]+]] = OpLoad %[[Ty_51]] %[[PtrId]]
; CHECK-NEXT:  %[[#]] = OpFixedExpINTEL %[[Ty_51]] %[[ExpInId2]] 0 20 20 0 0

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

; Function Attrs: norecurse
define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function() !kernel_arg_addr_space !{} !kernel_arg_access_qual !{} !kernel_arg_type !{} !kernel_arg_base_type !{} !kernel_arg_type_qual !{} {
entry:
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %0) 
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  call spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %1)
  call void @llvm.lifetime.end.p0(i64 1, ptr %0) 
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) 

; Function Attrs: inlinehint norecurse
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %this)  align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  call spir_func void @_Z4sqrtILi13ELi5ELb0ELi2ELi2EEvv()
  call spir_func void @_Z5recipILi3ELi8ELb1ELi4ELi4EEvv()
  call spir_func void @_Z5rsqrtILi11ELi10ELb0ELi8ELi6EEvv()
  call spir_func void @_Z3sinILi17ELi11ELb1ELi7ELi5EEvv()
  call spir_func void @_Z3cosILi35ELi28ELb0ELi9ELi3EEvv()
  call spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv()
  call spir_func void @_Z6sin_piILi60ELi5ELb0ELi2ELi2EEvv()
  call spir_func void @_Z6cos_piILi28ELi16ELb0ELi8ELi5EEvv()
  call spir_func void @_Z10sin_cos_piILi13ELi5ELb0ELi2ELi2EEvv()
  call spir_func void @_Z3logILi64ELi44ELb1ELi24ELi22EEvv()
  call spir_func void @_Z3expILi44ELi34ELb0ELi20ELi20EEvv()
  call spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv_()
  call spir_func void @_Z3expILi51ELi51ELb0ELi20ELi20EEvv()
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) 

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z4sqrtILi13ELi5ELb0ELi2ELi2EEvv()  {
entry:
  %a = alloca i13, align 2
  %ap_fixed_Sqrt = alloca i5, align 1
  %b = alloca i5, align 1
  %ap_fixed_Sqrt_b = alloca i13, align 2
  %c = alloca i5, align 1
  %ap_fixed_Sqrt_c = alloca i13, align 2
  call void @llvm.lifetime.start.p0(i64 2, ptr %a)
  call void @llvm.lifetime.start.p0(i64 1, ptr %ap_fixed_Sqrt)
  %0 = load i13, ptr %a, align 2
  %call = call spir_func signext i5 @_Z22__spirv_FixedSqrtINTELILi13ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i13 signext %0, i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
  store i5 %call, ptr %ap_fixed_Sqrt, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %b)
  call void @llvm.lifetime.start.p0(i64 2, ptr %ap_fixed_Sqrt_b)
  %1 = load i5, ptr %b, align 1
  %call1 = call spir_func signext i13 @_Z22__spirv_FixedSqrtINTELILi5ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i5 signext %1, i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
  store i13 %call1, ptr %ap_fixed_Sqrt_b, align 2
  call void @llvm.lifetime.start.p0(i64 1, ptr %c)
  call void @llvm.lifetime.start.p0(i64 2, ptr %ap_fixed_Sqrt_c)
  %2 = load i5, ptr %c, align 1
  %call2 = call spir_func signext i13 @_Z22__spirv_FixedSqrtINTELILi5ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i5 signext %2, i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
  store i13 %call2, ptr %ap_fixed_Sqrt_c, align 2
  call void @llvm.lifetime.end.p0(i64 2, ptr %ap_fixed_Sqrt_c)
  call void @llvm.lifetime.end.p0(i64 1, ptr %c)
  call void @llvm.lifetime.end.p0(i64 2, ptr %ap_fixed_Sqrt_b)
  call void @llvm.lifetime.end.p0(i64 1, ptr %b)
  call void @llvm.lifetime.end.p0(i64 1, ptr %ap_fixed_Sqrt)
  call void @llvm.lifetime.end.p0(i64 2, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z5recipILi3ELi8ELb1ELi4ELi4EEvv() {
entry:
  %a = alloca i3, align 1
  %ap_fixed_Recip = alloca i8, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %a)
  call void @llvm.lifetime.start.p0(i64 1, ptr %ap_fixed_Recip)
  %0 = load i3, ptr %a, align 1
  %call = call spir_func signext i8 @_Z23__spirv_FixedRecipINTELILi3ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i3 signext %0, i1 zeroext true, i32 4, i32 4, i32 0, i32 0)
  store i8 %call, ptr %ap_fixed_Recip, align 1
  call void @llvm.lifetime.end.p0(i64 1, ptr %ap_fixed_Recip)
  call void @llvm.lifetime.end.p0(i64 1, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z5rsqrtILi11ELi10ELb0ELi8ELi6EEvv() {
entry:
  %a = alloca i11, align 2
  %ap_fixed_Rsqrt = alloca i10, align 2
  call void @llvm.lifetime.start.p0(i64 2, ptr %a)
  call void @llvm.lifetime.start.p0(i64 2, ptr %ap_fixed_Rsqrt)
  %0 = load i11, ptr %a, align 2
  %call = call spir_func signext i10 @_Z23__spirv_FixedRsqrtINTELILi11ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i11 signext %0, i1 zeroext false, i32 8, i32 6, i32 0, i32 0)
  store i10 %call, ptr %ap_fixed_Rsqrt, align 2
  call void @llvm.lifetime.end.p0(i64 2, ptr %ap_fixed_Rsqrt)
  call void @llvm.lifetime.end.p0(i64 2, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3sinILi17ELi11ELb1ELi7ELi5EEvv() {
entry:
  %a = alloca i17, align 4
  %ap_fixed_Sin = alloca i11, align 2
  call void @llvm.lifetime.start.p0(i64 4, ptr %a)
  call void @llvm.lifetime.start.p0(i64 2, ptr %ap_fixed_Sin)
  %0 = load i17, ptr %a, align 4
  %call = call spir_func signext i11 @_Z21__spirv_FixedSinINTELILi17ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i17 signext %0, i1 zeroext true, i32 7, i32 5, i32 0, i32 0)
  store i11 %call, ptr %ap_fixed_Sin, align 2
  call void @llvm.lifetime.end.p0(i64 2, ptr %ap_fixed_Sin)
  call void @llvm.lifetime.end.p0(i64 4, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3cosILi35ELi28ELb0ELi9ELi3EEvv() {
entry:
  %a = alloca i35, align 8
  %ap_fixed_Cos = alloca i28, align 4
  call void @llvm.lifetime.start.p0(i64 8, ptr %a)
  call void @llvm.lifetime.start.p0(i64 4, ptr %ap_fixed_Cos)
  %0 = load i35, ptr %a, align 8
  %call = call spir_func signext i28 @_Z21__spirv_FixedCosINTELILi35ELi28EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i35 %0, i1 zeroext false, i32 9, i32 3, i32 0, i32 0)
  store i28 %call, ptr %ap_fixed_Cos, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr %ap_fixed_Cos)
  call void @llvm.lifetime.end.p0(i64 8, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv() {
entry:
  %a = alloca i31, align 4
  %ap_fixed_SinCos = alloca i40, align 8
  call void @llvm.lifetime.start.p0(i64 4, ptr %a)
  call void @llvm.lifetime.start.p0(i64 8, ptr %ap_fixed_SinCos)
  %0 = load i31, ptr %a, align 4
  %call = call spir_func i40 @_Z24__spirv_FixedSinCosINTELILi31ELi20EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i31 signext %0, i1 zeroext true, i32 10, i32 12, i32 0, i32 0)
  store i40 %call, ptr %ap_fixed_SinCos, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr %ap_fixed_SinCos)
  call void @llvm.lifetime.end.p0(i64 4, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z6sin_piILi60ELi5ELb0ELi2ELi2EEvv() {
entry:
  %a = alloca i60, align 8
  %ap_fixed_SinPi = alloca i5, align 1
  call void @llvm.lifetime.start.p0(i64 8, ptr %a)
  call void @llvm.lifetime.start.p0(i64 1, ptr %ap_fixed_SinPi)
  %0 = load i60, ptr %a, align 8
  %call = call spir_func signext i5 @_Z23__spirv_FixedSinPiINTELILi60ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i60 %0, i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
  store i5 %call, ptr %ap_fixed_SinPi, align 1
  call void @llvm.lifetime.end.p0(i64 1, ptr %ap_fixed_SinPi)
  call void @llvm.lifetime.end.p0(i64 8, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z6cos_piILi28ELi16ELb0ELi8ELi5EEvv() {
entry:
  %a = alloca i28, align 4
  %ap_fixed_CosPi = alloca i16, align 2
  call void @llvm.lifetime.start.p0(i64 4, ptr %a)
  call void @llvm.lifetime.start.p0(i64 2, ptr %ap_fixed_CosPi)
  %0 = load i28, ptr %a, align 4
  %call = call spir_func signext i16 @_Z23__spirv_FixedCosPiINTELILi28ELi16EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i28 signext %0, i1 zeroext false, i32 8, i32 5, i32 0, i32 0)
  store i16 %call, ptr %ap_fixed_CosPi, align 2
  call void @llvm.lifetime.end.p0(i64 2, ptr %ap_fixed_CosPi)
  call void @llvm.lifetime.end.p0(i64 4, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z10sin_cos_piILi13ELi5ELb0ELi2ELi2EEvv() {
entry:
  %a = alloca i13, align 2
  %ap_fixed_SinCosPi = alloca i10, align 2
  call void @llvm.lifetime.start.p0(i64 2, ptr %a)
  call void @llvm.lifetime.start.p0(i64 2, ptr %ap_fixed_SinCosPi)
  %0 = load i13, ptr %a, align 2
  %call = call spir_func signext i10 @_Z26__spirv_FixedSinCosPiINTELILi13ELi5EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i13 signext %0, i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
  store i10 %call, ptr %ap_fixed_SinCosPi, align 2
  call void @llvm.lifetime.end.p0(i64 2, ptr %ap_fixed_SinCosPi)
  call void @llvm.lifetime.end.p0(i64 2, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3logILi64ELi44ELb1ELi24ELi22EEvv() {
entry:
  %a = alloca i64, align 8
  %ap_fixed_Log = alloca i44, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr %a)
  call void @llvm.lifetime.start.p0(i64 8, ptr %ap_fixed_Log)
  %0 = load i64, ptr %a, align 8
  %call = call spir_func i44 @_Z21__spirv_FixedLogINTELILi64ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i64 %0, i1 zeroext true, i32 24, i32 22, i32 0, i32 0)
  store i44 %call, ptr %ap_fixed_Log, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr %ap_fixed_Log)
  call void @llvm.lifetime.end.p0(i64 8, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3expILi44ELi34ELb0ELi20ELi20EEvv() {
entry:
  %a = alloca i44, align 8
  %ap_fixed_Exp = alloca i34, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr %a)
  call void @llvm.lifetime.start.p0(i64 8, ptr %ap_fixed_Exp)
  %0 = load i44, ptr %a, align 8
  %call = call spir_func i34 @_Z21__spirv_FixedExpINTELILi44ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i44 %0, i1 zeroext false, i32 20, i32 20, i32 0, i32 0)
  store i34 %call, ptr %ap_fixed_Exp, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr %ap_fixed_Exp)
  call void @llvm.lifetime.end.p0(i64 8, ptr %a)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv_() {
entry:
  %0 = alloca i34, align 8
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = alloca i51, align 8
  %3 = addrspacecast ptr %2 to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 8, ptr %0)
  call void @llvm.lifetime.start.p0(i64 16, ptr %2)
  %4 = load i34, ptr addrspace(4) %1, align 8
  call spir_func void @_Z24__spirv_FixedSinCosINTELILi34ELi51EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(ptr addrspace(4) sret(i51) align 8 %3, i34 %4, i1 zeroext true, i32 3, i32 2, i32 0, i32 0)
  %5 = load i51, ptr addrspace(4) %3, align 8
  store i51 %5, ptr addrspace(4) %3, align 8
  call void @llvm.lifetime.end.p0(i64 16, ptr %2)
  call void @llvm.lifetime.end.p0(i64 8, ptr %0)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3expILi51ELi51ELb0ELi20ELi20EEvv() {
entry:
  %a = alloca i51, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %ap_fixed_Exp = alloca i51, align 8
  %ap_fixed_Exp.ascast = addrspacecast ptr %ap_fixed_Exp to ptr addrspace(4)
  %tmp = alloca i51, align 8
  %tmp.ascast = addrspacecast ptr %tmp to ptr addrspace(4)
  %indirect-arg-temp = alloca i51, align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr %a)
  call void @llvm.lifetime.start.p0(i64 16, ptr %ap_fixed_Exp)
  %0 = load i51, ptr addrspace(4) %a.ascast, align 8
  store i51 %0, ptr %indirect-arg-temp, align 8
  call spir_func void @_Z21__spirv_FixedExpINTELILi51ELi51EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(ptr addrspace(4) sret(i51) align 8 %tmp.ascast, ptr byval(i64) align 8 %indirect-arg-temp, i1 zeroext false, i32 20, i32 20, i32 0, i32 0)
  %1 = load i51, ptr addrspace(4) %tmp.ascast, align 8
  store i51 %1, ptr addrspace(4) %ap_fixed_Exp.ascast, align 8
  call void @llvm.lifetime.end.p0(i64 16, ptr %ap_fixed_Exp)
  call void @llvm.lifetime.end.p0(i64 16, ptr %a)
  ret void
}


; Function Attrs: nounwind
declare dso_local spir_func signext i5 @_Z22__spirv_FixedSqrtINTELILi13ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i13 signext, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func signext i13 @_Z22__spirv_FixedSqrtINTELILi5ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i5 signext, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func signext i8 @_Z23__spirv_FixedRecipINTELILi3ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i3 signext, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func signext i10 @_Z23__spirv_FixedRsqrtINTELILi11ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i11 signext, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func signext i11 @_Z21__spirv_FixedSinINTELILi17ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i17 signext, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func signext i28 @_Z21__spirv_FixedCosINTELILi35ELi28EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i35, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func i40 @_Z24__spirv_FixedSinCosINTELILi31ELi20EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i31 signext, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func signext i5 @_Z23__spirv_FixedSinPiINTELILi60ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i60, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func signext i16 @_Z23__spirv_FixedCosPiINTELILi28ELi16EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i28 signext, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func signext i10 @_Z26__spirv_FixedSinCosPiINTELILi13ELi5EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i13 signext, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func i44 @_Z21__spirv_FixedLogINTELILi64ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i64, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func i34 @_Z21__spirv_FixedExpINTELILi44ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i44, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: nounwind
declare dso_local spir_func void @_Z24__spirv_FixedSinCosINTELILi34ELi51EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(ptr addrspace(4) sret(i51) align 8, i34, i1 zeroext, i32, i32, i32, i32)

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z21__spirv_FixedExpINTELILi51ELi51EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(ptr addrspace(4) sret(i51) align 8, ptr byval(i51) align 8, i1 zeroext, i32, i32, i32, i32)

