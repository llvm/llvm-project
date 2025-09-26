; RUN: llc -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s 
; TODO: %if spirv-tools %{ llc -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Kernel
; CHECK: OpCapability ArbitraryPrecisionFloatingPointINTEL
; CHECK: OpCapability ArbitraryPrecisionIntegersINTEL
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_floating_point"
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_integers"

; CHECK-DAG: OpTypeInt 2 0
; CHECK-DAG: OpTypeInt 13 0
; CHECK-DAG: OpTypeInt 14 0
; CHECK-DAG: OpTypeInt 17 0
; CHECK-DAG: OpTypeInt 18 0
; CHECK-DAG: OpTypeInt 25 0
; CHECK-DAG: OpTypeInt 30 0
; CHECK-DAG: OpTypeInt 34 0
; CHECK-DAG: OpTypeInt 39 0
; CHECK-DAG: OpTypeInt 40 0
; CHECK-DAG: OpTypeInt 42 0
; CHECK-DAG: OpTypeInt 51 0
; CHECK-DAG: OpTypeBool

; CHECK: OpFunction
; CHECK: OpLabel
; CHECK: OpArbitraryFloatCastINTEL 
; CHECK: OpArbitraryFloatCastFromIntINTEL 
; CHECK: OpArbitraryFloatCastToIntINTEL 
; CHECK: OpArbitraryFloatAddINTEL 
; CHECK: OpArbitraryFloatSubINTEL 
; CHECK: OpArbitraryFloatMulINTEL 
; CHECK: OpArbitraryFloatDivINTEL 
; CHECK: OpArbitraryFloatGTINTEL 
; CHECK: OpArbitraryFloatGEINTEL 
; CHECK: OpArbitraryFloatLTINTEL 
; CHECK: OpArbitraryFloatLEINTEL 
; CHECK: OpArbitraryFloatEQINTEL 
; CHECK: OpArbitraryFloatRecipINTEL 
; CHECK: OpArbitraryFloatRSqrtINTEL 
; CHECK: OpArbitraryFloatCbrtINTEL 
; CHECK: OpArbitraryFloatHypotINTEL 
; CHECK: OpArbitraryFloatSqrtINTEL 

define dso_local spir_kernel void @test() {
entry:
  %0 = alloca i63, align 8
  %1 = alloca i63, align 8
  %2 = alloca i8, align 1
  call spir_func void @ap_float_ops(i63* %0, i63* %1, i8* %2)
  ret void
}

define internal spir_func void @ap_float_ops(i63* %in1, i63* %in2, i8* %out)  {
entry:
  %0 = load i63, ptr %in1, align 8
  %1 = load i63, ptr %in2, align 8
  %cast = call spir_func i40 @_Z31__spirv_ArbitraryFloatCastINTELILi63ELi40EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i63 %0, i32 28, i32 30, i32 0, i32 2, i32 1)
  %cast_from_int = call spir_func signext i25 @_Z38__spirv_ArbitraryFloatCastFromIntINTELILi40ELi25EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiibiii(i40 %cast, i32 16, i1 zeroext false, i32 0, i32 2, i32 1)
  %cast_to_int = call spir_func signext i30 @_Z36__spirv_ArbitraryFloatCastToIntINTELILi25ELi30EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiibiii(i25 signext %cast_from_int, i32 15, i1 zeroext true, i32 0, i32 2, i32 1)
  %add = call spir_func signext i14 @_Z30__spirv_ArbitraryFloatAddINTELILi30ELi40ELi14EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i30 signext %cast_to_int, i32 7, i40 %cast, i32 8, i32 9, i32 0, i32 2, i32 1)
  %sub = call spir_func signext i13 @_Z30__spirv_ArbitraryFloatSubINTELILi14ELi30ELi13EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i14 signext %add, i32 4, i30 signext %cast_to_int, i32 5, i32 6, i32 0, i32 2, i32 1)
  %mul = call spir_func i51 @_Z30__spirv_ArbitraryFloatMulINTELILi63ELi63ELi51EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i63 %0, i32 34, i63 %1, i32 34, i32 34, i32 0, i32 2, i32 1)
  %div = call spir_func signext i18 @_Z30__spirv_ArbitraryFloatDivINTELILi51ELi40ELi18EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i51 signext %mul, i32 11, i40 signext %cast, i32 11, i32 12, i32 0, i32 2, i32 1)
  %gt = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatGTINTELILi63ELi63EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i63 %0, i32 42, i63 %1, i32 41)
  %ge = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatGEINTELILi51ELi40EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i51 %mul, i32 27, i40 %cast, i32 27)
  %lt = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatLTINTELILi14ELi30EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i14 signext %add, i32 2, i30 signext %cast_to_int, i32 3)
  %le = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatLEINTELILi51ELi40EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i51 %mul, i32 27, i40 %cast, i32 28)
  %eq = call spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatEQINTELILi18ELi14EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i18 signext %div, i32 12, i14 signext %add, i32 7)
  %recip = call spir_func i39 @_Z32__spirv_ArbitraryFloatRecipINTELILi40ELi39EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i40 %cast, i32 29, i32 29, i32 0, i32 2, i32 1)
  %rsqrt = call spir_func i34 @_Z32__spirv_ArbitraryFloatRSqrtINTELILi39ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i39 %recip, i32 19, i32 20, i32 0, i32 2, i32 1)
  %cbrt = call spir_func signext i2 @_Z31__spirv_ArbitraryFloatCbrtINTELILi13ELi2EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i13 signext %sub, i32 1, i32 1, i32 0, i32 2, i32 1)
  %hypot = call spir_func i42 @_Z32__spirv_ArbitraryFloatHypotINTELILi40ELi40ELi42EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i40 %cast, i32 20, i40 %cast, i32 21, i32 22, i32 0, i32 2, i32 1)
  %sqrt = call spir_func signext i17 @_Z31__spirv_ArbitraryFloatSqrtINTELILi14ELi17EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i14 signext %add, i32 7, i32 8, i32 0, i32 2, i32 1)
  %sin = call spir_func signext i17 @_Z31__spirv_ArbitraryFloatSinINTELILi17ELi17EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i17 signext %sqrt, i32 7, i32 8, i32 0, i32 2, i32 1)
  %final = zext i1 %gt to i8
  store i8 %final, ptr %out, align 1
  ret void
}
declare dso_local spir_func i40 @_Z31__spirv_ArbitraryFloatCastINTELILi63ELi40EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i63, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i25 @_Z38__spirv_ArbitraryFloatCastFromIntINTELILi40ELi25EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiibiii(i40, i32, i1 zeroext, i32, i32, i32) 
declare dso_local spir_func signext i30 @_Z36__spirv_ArbitraryFloatCastToIntINTELILi25ELi30EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiibiii(i25 signext, i32, i1 zeroext, i32, i32, i32) 
declare dso_local spir_func signext i14 @_Z30__spirv_ArbitraryFloatAddINTELILi30ELi40ELi14EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i30 signext, i32, i40, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i13 @_Z30__spirv_ArbitraryFloatSubINTELILi14ELi30ELi13EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i14 signext, i32, i30 signext, i32, i32, i32, i32, i32)
declare dso_local spir_func i51 @_Z30__spirv_ArbitraryFloatMulINTELILi63ELi63ELi51EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i63, i32, i63, i32, i32, i32, i32, i32)
declare dso_local spir_func signext i18 @_Z30__spirv_ArbitraryFloatDivINTELILi51ELi40ELi18EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i51 signext, i32, i40 signext, i32, i32, i32, i32, i32)
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatGTINTELILi63ELi63EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i63, i32, i63, i32) 
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatGEINTELILi51ELi40EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i51, i32, i40, i32)
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatLTINTELILi14ELi30EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i14 signext, i32, i30 signext, i32)
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatLEINTELILi51ELi40EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i51, i32, i40, i32)
declare dso_local spir_func zeroext i1 @_Z29__spirv_ArbitraryFloatEQINTELILi18ELi14EEbU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEii(i18 signext, i32, i14 signext, i32)
declare dso_local spir_func i39 @_Z32__spirv_ArbitraryFloatRecipINTELILi40ELi39EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i40, i32, i32, i32, i32, i32) 
declare dso_local spir_func i34 @_Z32__spirv_ArbitraryFloatRSqrtINTELILi39ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i39, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i2 @_Z31__spirv_ArbitraryFloatCbrtINTELILi13ELi2EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i13 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func i42 @_Z32__spirv_ArbitraryFloatHypotINTELILi40ELi40ELi42EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i40, i32, i40, i32, i32, i32, i32, i32)
declare dso_local spir_func signext i17 @_Z31__spirv_ArbitraryFloatSqrtINTELILi14ELi17EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i14 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i17 @_Z31__spirv_ArbitraryFloatSinINTELILi17ELi17EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i17 signext, i32, i32, i32, i32, i32) 
