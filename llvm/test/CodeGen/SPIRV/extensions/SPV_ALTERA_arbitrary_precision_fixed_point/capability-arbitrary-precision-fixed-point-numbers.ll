; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_fixed_point,+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s 
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_fixed_point,+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Kernel
; CHECK-DAG: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK-DAG: OpCapability ArbitraryPrecisionFixedPointALTERA
; CHECK-DAG: OpExtension "SPV_ALTERA_arbitrary_precision_fixed_point"
; CHECK-DAG: OpExtension "SPV_ALTERA_arbitrary_precision_integers"

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
; CHECK-NEXT:  %[[#]] = OpFixedSqrtALTERA %[[Ty_5]] %[[Sqrt_InId]] 0 2 2 0 0

; CHECK:        %[[Recip_InId:[0-9]+]] = OpLoad %[[Ty_3]]
; CHECK-NEXT:  %[[#]] = OpFixedRecipALTERA %[[Ty_8]] %[[Recip_InId]] 1 4 4 0 0

; CHECK:        %[[Rsqrt_InId:[0-9]+]] = OpLoad %[[Ty_11]]
; CHECK-NEXT:  %[[#]] = OpFixedRsqrtALTERA %[[Ty_10]] %[[Rsqrt_InId]] 0 8 6 0 0

; CHECK:        %[[Sin_InId:[0-9]+]] = OpLoad %[[Ty_17]]
; CHECK-NEXT:  %[[#]] = OpFixedSinALTERA %[[Ty_11]] %[[Sin_InId]] 1 7 5 0 0

; CHECK:        %[[Cos_InId:[0-9]+]] = OpLoad %[[Ty_35]]
; CHECK-NEXT:  %[[#]] = OpFixedCosALTERA %[[Ty_28]] %[[Cos_InId]] 0 9 3 0 0

; CHECK:        %[[SinCos_InId:[0-9]+]] = OpLoad %[[Ty_31]]
; CHECK-NEXT:  %[[#]] = OpFixedSinCosALTERA %[[Ty_40]] %[[SinCos_InId]] 1 10 12 0 0

; CHECK:        %[[SinPi_InId:[0-9]+]] = OpLoad %[[Ty_60]]
; CHECK-NEXT:  %[[#]] = OpFixedSinPiALTERA %[[Ty_5]] %[[SinPi_InId]] 0 2 2 0 0

; CHECK:        %[[CosPi_InId:[0-9]+]] = OpLoad %[[Ty_28]]
; CHECK-NEXT:  %[[#]] = OpFixedCosPiALTERA %[[Ty_16]] %[[CosPi_InId]] 0 8 5 0 0

; CHECK:        %[[SinCosPi_InId:[0-9]+]] = OpLoad %[[Ty_13]]
; CHECK-NEXT:  %[[#]] = OpFixedSinCosPiALTERA %[[Ty_10]] %[[SinCosPi_InId]] 0 2 2 0 0

; CHECK:        %[[Log_InId:[0-9]+]] = OpLoad %[[Ty_64]]
; CHECK-NEXT:  %[[#]] = OpFixedLogALTERA %[[Ty_44]] %[[Log_InId]] 1 24 22 0 0

; CHECK:        %[[Exp_InId:[0-9]+]] = OpLoad %[[Ty_44]]
; CHECK-NEXT:  %[[#]] = OpFixedExpALTERA %[[Ty_34]] %[[Exp_InId]] 0 20 20 0 0

; CHECK:        %[[SinCos_InId:[0-9]+]] = OpLoad %[[Ty_34]]
; CHECK-NEXT:  %[[SinCos_ResultId:[0-9]+]] = OpFixedSinCosALTERA %[[Ty_51]] %[[SinCos_InId]] 1 3 2 0 0
; CHECK-NEXT:        OpStore %[[#]] %[[SinCos_ResultId]]

; CHECK:       %[[ResId:[0-9]+]] = OpLoad %[[Ty_51]]
; CHECK-NEXT:  OpStore %[[PtrId:[0-9]+]] %[[ResId]]
; CHECK-NEXT:  %[[ExpInId2:[0-9]+]] = OpLoad %[[Ty_51]] %[[PtrId]]
; CHECK-NEXT:  %[[#]] = OpFixedExpALTERA %[[Ty_51]] %[[ExpInId2]] 0 20 20 0 0

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function() !kernel_arg_addr_space !{} !kernel_arg_access_qual !{} !kernel_arg_type !{} !kernel_arg_base_type !{} !kernel_arg_type_qual !{} {
entry:
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  call spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %1)
  ret void
}

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

define linkonce_odr dso_local spir_func void @_Z4sqrtILi13ELi5ELb0ELi2ELi2EEvv() {
entry:
  %in_ptr  = alloca i13, align 2
  %out_ptr = alloca i5,  align 1
  %in_val  = load i13, ptr %in_ptr, align 2
  %res     = call spir_func signext i5 @_Z22__spirv_FixedSqrtINTELILi13ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i13 signext %in_val, i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
  store i5 %res, ptr %out_ptr, align 1
  ret void
}

define linkonce_odr dso_local spir_func void @_Z5recipILi3ELi8ELb1ELi4ELi4EEvv() {
entry:
  %in_ptr  = alloca i3, align 1
  %out_ptr = alloca i8, align 1
  %in_val  = load i3, ptr %in_ptr, align 1
  %res     = call spir_func signext i8 @_Z23__spirv_FixedRecipINTELILi3ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i3 signext %in_val, i1 zeroext true, i32 4, i32 4, i32 0, i32 0)
  store i8 %res, ptr %out_ptr, align 1
  ret void
}

define linkonce_odr dso_local spir_func void @_Z5rsqrtILi11ELi10ELb0ELi8ELi6EEvv() {
entry:
  %in_ptr  = alloca i11, align 2
  %out_ptr = alloca i10, align 2
  %in_val  = load i11, ptr %in_ptr, align 2
  %res     = call spir_func signext i10 @_Z23__spirv_FixedRsqrtINTELILi11ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i11 signext %in_val, i1 zeroext false, i32 8, i32 6, i32 0, i32 0)
  store i10 %res, ptr %out_ptr, align 2
  ret void
}

define linkonce_odr dso_local spir_func void @_Z3sinILi17ELi11ELb1ELi7ELi5EEvv() {
entry:
  %in_ptr  = alloca i17, align 4
  %out_ptr = alloca i11, align 2
  %in_val  = load i17, ptr %in_ptr, align 4
  %res     = call spir_func signext i11 @_Z21__spirv_FixedSinINTELILi17ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i17 signext %in_val, i1 zeroext true, i32 7, i32 5, i32 0, i32 0)
  store i11 %res, ptr %out_ptr, align 2
  ret void
}

define linkonce_odr dso_local spir_func void @_Z3cosILi35ELi28ELb0ELi9ELi3EEvv() {
entry:
  %in_ptr  = alloca i35, align 8
  %out_ptr = alloca i28, align 4
  %in_val  = load i35, ptr %in_ptr, align 8
  %res     = call spir_func signext i28 @_Z21__spirv_FixedCosINTELILi35ELi28EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i35 signext %in_val, i1 zeroext false, i32 9, i32 3, i32 0, i32 0)
  store i28 %res, ptr %out_ptr, align 4
  ret void
}

define linkonce_odr dso_local spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv() {
entry:
  %in_ptr  = alloca i31, align 4
  %out_ptr = alloca i40, align 8
  %in_val  = load i31, ptr %in_ptr, align 4
  %res     = call spir_func i40 @_Z24__spirv_FixedSinCosINTELILi31ELi20EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i31 signext %in_val, i1 zeroext true, i32 10, i32 12, i32 0, i32 0)
  store i40 %res, ptr %out_ptr, align 8
  ret void
}

define linkonce_odr dso_local spir_func void @_Z6sin_piILi60ELi5ELb0ELi2ELi2EEvv() {
entry:
  %in_ptr  = alloca i60, align 8
  %out_ptr = alloca i5,  align 1
  %in_val  = load i60, ptr %in_ptr, align 8
  %res     = call spir_func signext i5 @_Z23__spirv_FixedSinPiINTELILi60ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i60 signext %in_val, i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
  store i5 %res, ptr %out_ptr, align 1
  ret void
}

define linkonce_odr dso_local spir_func void @_Z6cos_piILi28ELi16ELb0ELi8ELi5EEvv() {
entry:
  %in_ptr  = alloca i28, align 4
  %out_ptr = alloca i16, align 2
  %in_val  = load i28, ptr %in_ptr, align 4
  %res     = call spir_func signext i16 @_Z23__spirv_FixedCosPiINTELILi28ELi16EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i28 signext %in_val, i1 zeroext false, i32 8, i32 5, i32 0, i32 0)
  store i16 %res, ptr %out_ptr, align 2
  ret void
}

define linkonce_odr dso_local spir_func void @_Z10sin_cos_piILi13ELi5ELb0ELi2ELi2EEvv() {
entry:
  %in_ptr  = alloca i13, align 2
  %out_ptr = alloca i10, align 2
  %in_val  = load i13, ptr %in_ptr, align 2
  %res     = call spir_func signext i10 @_Z26__spirv_FixedSinCosPiINTELILi13ELi5EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i13 signext %in_val, i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
  store i10 %res, ptr %out_ptr, align 2
  ret void
}

define linkonce_odr dso_local spir_func void @_Z3logILi64ELi44ELb1ELi24ELi22EEvv() {
entry:
  %in_ptr  = alloca i64, align 8
  %out_ptr = alloca i44, align 8
  %in_val  = load i64, ptr %in_ptr, align 8
  %res     = call spir_func i44 @_Z21__spirv_FixedLogINTELILi64ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i64 %in_val, i1 zeroext true, i32 24, i32 22, i32 0, i32 0)
  store i44 %res, ptr %out_ptr, align 8
  ret void
}

define linkonce_odr dso_local spir_func void @_Z3expILi44ELi34ELb0ELi20ELi20EEvv() {
entry:
  %in_ptr  = alloca i44, align 8
  %out_ptr = alloca i34, align 8
  %in_val  = load i44, ptr %in_ptr, align 8
  %res     = call spir_func i34 @_Z21__spirv_FixedExpINTELILi44ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i44 %in_val, i1 zeroext false, i32 20, i32 20, i32 0, i32 0)
  store i34 %res, ptr %out_ptr, align 8
  ret void
}

define linkonce_odr dso_local spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv_() {
entry:
  %tmp     = alloca i34, align 8
  %out_ptr = alloca i51, align 8
  %in_ptr  = addrspacecast ptr %tmp to ptr addrspace(4)
  %out_s   = addrspacecast ptr %out_ptr to ptr addrspace(4)
  %in_val  = load i34, ptr addrspace(4) %in_ptr, align 8
  call spir_func void @_Z24__spirv_FixedSinCosINTELILi34ELi51EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(ptr addrspace(4) sret(i51) align 8 %out_s, i34 %in_val, i1 zeroext true, i32 3, i32 2, i32 0, i32 0)
  ret void
}

define linkonce_odr dso_local spir_func void @_Z3expILi51ELi51ELb0ELi20ELi20EEvv() {
entry:
  %a = alloca i51, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %ap_fixed_Exp = alloca i51, align 8
  %ap_fixed_Exp.ascast = addrspacecast ptr %ap_fixed_Exp to ptr addrspace(4)
  %tmp = alloca i51, align 8
  %tmp.ascast = addrspacecast ptr %tmp to ptr addrspace(4)
  %indirect-arg-temp = alloca i51, align 8
  %0 = load i51, ptr addrspace(4) %a.ascast, align 8
  store i51 %0, ptr %indirect-arg-temp, align 8
  call spir_func void @_Z21__spirv_FixedExpINTELILi51ELi51EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(
      ptr addrspace(4) sret(i51) align 8 %tmp.ascast,
      ptr byval(i64) align 8 %indirect-arg-temp,
      i1 zeroext false, i32 20, i32 20, i32 0, i32 0)
  %1 = load i51, ptr addrspace(4) %tmp.ascast, align 8
  store i51 %1, ptr addrspace(4) %ap_fixed_Exp.ascast, align 8
  ret void
}

declare dso_local spir_func signext i5 @_Z22__spirv_FixedSqrtINTELILi13ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i13 signext, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func signext i13 @_Z22__spirv_FixedSqrtINTELILi5ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i5 signext, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func signext i8 @_Z23__spirv_FixedRecipINTELILi3ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i3 signext, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func signext i10 @_Z23__spirv_FixedRsqrtINTELILi11ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i11 signext, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func signext i11 @_Z21__spirv_FixedSinINTELILi17ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i17 signext, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func signext i28 @_Z21__spirv_FixedCosINTELILi35ELi28EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i35, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func i40 @_Z24__spirv_FixedSinCosINTELILi31ELi20EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i31 signext, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func signext i5 @_Z23__spirv_FixedSinPiINTELILi60ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i60, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func signext i16 @_Z23__spirv_FixedCosPiINTELILi28ELi16EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i28 signext, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func signext i10 @_Z26__spirv_FixedSinCosPiINTELILi13ELi5EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i13 signext, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func i44 @_Z21__spirv_FixedLogINTELILi64ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i64, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func i34 @_Z21__spirv_FixedExpINTELILi44ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i44, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func void @_Z24__spirv_FixedSinCosINTELILi34ELi51EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(ptr addrspace(4) sret(i51) align 8, i34, i1 zeroext, i32, i32, i32, i32)
declare dso_local spir_func void @_Z21__spirv_FixedExpINTELILi51ELi51EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(ptr addrspace(4) sret(i51) align 8, ptr byval(i51) align 8, i1 zeroext, i32, i32, i32, i32)
