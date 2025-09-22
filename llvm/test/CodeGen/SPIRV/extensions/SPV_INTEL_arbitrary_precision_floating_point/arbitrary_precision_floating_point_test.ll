; RUN: llc -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s 
; TODO: %if spirv-tools %{ llc -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Kernel
; CHECK: OpCapability ArbitraryPrecisionFloatingPointINTEL
; CHECK: OpCapability ArbitraryPrecisionIntegersINTEL
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_floating_point"
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_integers"

; CHECK-DAG: OpTypeInt 4 0
; CHECK-DAG: OpTypeInt 50 0
; CHECK-DAG: OpTypeInt 56 0
; CHECK-DAG: OpTypeInt 59 0
; CHECK-DAG: OpTypeInt 5 0
; CHECK-DAG: OpTypeInt 8 0
; CHECK-DAG: OpTypeInt 10 0
; CHECK-DAG: OpTypeInt 11 0
; CHECK-DAG: OpTypeInt 13 0
; CHECK-DAG: OpTypeInt 14 0
; CHECK-DAG: OpTypeInt 15 0
; CHECK-DAG: OpTypeInt 21 0
; CHECK-DAG: OpTypeInt 25 0
; CHECK-DAG: OpTypeInt 27 0
; CHECK-DAG: OpTypeInt 34 0
; CHECK-DAG: OpTypeInt 35 0
; CHECK-DAG: OpTypeInt 38 0
; CHECK-DAG: OpTypeInt 42 0
; CHECK-DAG: OpTypeInt 44 0
; CHECK-DAG: OpTypeInt 49 0
; CHECK-DAG: OpTypeInt 62 0
; CHECK-DAG: OpTypeInt 64 0
; CHECK-DAG: OpTypeBool

; CHECK: OpLabel
; CHECK: OpArbitraryFloatLogINTEL 
; CHECK: OpArbitraryFloatLog2INTEL 
; CHECK: OpArbitraryFloatLog10INTEL 
; CHECK: OpArbitraryFloatLog1pINTEL 
; CHECK: OpArbitraryFloatExpINTEL 
; CHECK: OpArbitraryFloatExp2INTEL 
; CHECK: OpArbitraryFloatExp10INTEL 
; CHECK: OpArbitraryFloatExpm1INTEL 
; CHECK: OpArbitraryFloatSinINTEL 
; CHECK: OpArbitraryFloatCosINTEL 
; CHECK: OpArbitraryFloatSinCosINTEL 
; CHECK: OpArbitraryFloatSinPiINTEL 
; CHECK: OpArbitraryFloatCosPiINTEL 
; CHECK: OpArbitraryFloatSinCosPiINTEL 
; CHECK: OpArbitraryFloatASinINTEL 
; CHECK: OpArbitraryFloatASinPiINTEL 
; CHECK: OpArbitraryFloatACosINTEL 
; CHECK: OpArbitraryFloatACosPiINTEL 
; CHECK: OpArbitraryFloatATanINTEL 
; CHECK: OpArbitraryFloatATanPiINTEL 
; CHECK: OpArbitraryFloatATan2INTEL 
; CHECK: OpArbitraryFloatPowINTEL 
; CHECK: OpArbitraryFloatPowRINTEL 
; CHECK: OpArbitraryFloatPowNINTEL 
; CHECK: OpArbitraryFloatPowNINTEL 

define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function() {
entry:
  %0 = alloca i64, align 8
  %1 = alloca i64, align 8
  %2 = alloca i8, align 1
  call spir_func void @ap_float_ops(i64* %0, i64* %1, i8* %2)
  ret void
}

define internal spir_func void @ap_float_ops(i64* %in1, i64* %in2, i8* %out)  {
entry:
  %0 = load i64, ptr %in1, align 8
  %1 = load i64, ptr %in2, align 8
  %log = call spir_func i50 @_Z30__spirv_ArbitraryFloatLogINTELILi50ELi50EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i64 %0, i32 19, i32 30, i32 0, i32 2, i32 1)
  %log2 = call spir_func i38 @_Z31__spirv_ArbitraryFloatLog2INTELILi38ELi38EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i50 %log, i32 20, i32 19, i32 0, i32 2, i32 1)
  %log10 = call spir_func signext i10 @_Z32__spirv_ArbitraryFloatLog10INTELILi8ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i38 %log2, i32 3, i32 5, i32 0, i32 2, i32 1)
  %log1p = call spir_func i49 @_Z32__spirv_ArbitraryFloatLog1pINTELILi48ELi49EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i50 %log, i32 30, i32 30, i32 0, i32 2, i32 1)
  %exp = call spir_func i42 @_Z30__spirv_ArbitraryFloatExpINTELILi42ELi42EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i49 %log1p, i32 25, i32 25, i32 0, i32 2, i32 1)
  %exp2 = call spir_func signext i5 @_Z31__spirv_ArbitraryFloatExp2INTELILi3ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i10 %log10, i32 1, i32 2, i32 0, i32 2, i32 1)
  %exp10 = call spir_func signext i25 @_Z32__spirv_ArbitraryFloatExp10INTELILi25ELi25EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i42 %exp, i32 16, i32 16, i32 0, i32 2, i32 1)
  %expm1 = call spir_func i62 @_Z32__spirv_ArbitraryFloatExpm1INTELILi64ELi62EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i64 %0, i32 42, i32 41, i32 0, i32 2, i32 1)
  %sin = call spir_func i34 @_Z30__spirv_ArbitraryFloatSinINTELILi30ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i25 %exp10, i32 15, i32 17, i32 0, i32 2, i32 1)
  %cos = call spir_func signext i4 @_Z30__spirv_ArbitraryFloatCosINTELILi4ELi4EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i5 %exp2, i32 2, i32 1, i32 0, i32 2, i32 1)
  %sincos = call spir_func i62 @_Z33__spirv_ArbitraryFloatSinCosINTELILi27ELi31EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(i34 %sin, i32 18, i32 20, i32 0, i32 2, i32 1)
  %sinpi = call spir_func signext i13 @_Z32__spirv_ArbitraryFloatSinPiINTELILi10ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i10 %log10, i32 6, i32 6, i32 0, i32 2, i32 1)
  %cospi = call spir_func i59 @_Z32__spirv_ArbitraryFloatCosPiINTELILi59ELi59EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i62 %expm1, i32 40, i32 40, i32 0, i32 2, i32 1)
  %sincospi = call spir_func i64 @_Z35__spirv_ArbitraryFloatSinCosPiINTELILi30ELi32EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(i34 %sin, i32 20, i32 20, i32 0, i32 2, i32 1)
  %asin = call spir_func signext i11 @_Z31__spirv_ArbitraryFloatASinINTELILi7ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i4 %cos, i32 4, i32 8, i32 0, i32 2, i32 1)
  %asinpi = call spir_func i35 @_Z33__spirv_ArbitraryFloatASinPiINTELILi35ELi35EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i49 %log1p, i32 23, i32 23, i32 0, i32 2, i32 1)
  %acos = call spir_func signext i14 @_Z31__spirv_ArbitraryFloatACosINTELILi14ELi14EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i13 %sinpi, i32 9, i32 10, i32 0, i32 2, i32 1)
  %acospi = call spir_func signext i8 @_Z33__spirv_ArbitraryFloatACosPiINTELILi8ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i38 %log2, i32 5, i32 4, i32 0, i32 2, i32 1)
  %atan = call spir_func i44 @_Z31__spirv_ArbitraryFloatATanINTELILi44ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i59 %cospi, i32 31, i32 31, i32 0, i32 2, i32 1)
  %atanpi = call spir_func i34 @_Z33__spirv_ArbitraryFloatATanPiINTELILi40ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i50 %log, i32 38, i32 32, i32 0, i32 2, i32 1)
  %atan2 = call spir_func signext i27 @_Z32__spirv_ArbitraryFloatATan2INTELILi24ELi25ELi27EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i25 %exp10, i32 16, i42 %exp, i32 17, i32 18, i32 0, i32 2, i32 1)
  %pow = call spir_func signext i21 @_Z30__spirv_ArbitraryFloatPowINTELILi17ELi19ELi21EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i34 %sin, i32 8, i38 %log2, i32 9, i32 10, i32 0, i32 2, i32 1)
  %powr = call spir_func i56 @_Z31__spirv_ArbitraryFloatPowRINTELILi54ELi55ELi56EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i59 %cospi, i32 35, i49 %log1p, i32 35, i32 35, i32 0, i32 2, i32 1)
  %pown = call spir_func signext i15 @_Z31__spirv_ArbitraryFloatPowNINTELILi12ELi10ELi15EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiii(i14 %acos, i32 7, i10 %log10, i1 zeroext false, i32 9, i32 0, i32 2, i32 1)
  %pown2 = call spir_func signext i15 @_Z31__spirv_ArbitraryFloatPowNINTELILi64ELi10ELi15EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiii(i64 %0, i32 7, i10 %log10, i1 zeroext true, i32 9, i32 0, i32 2, i32 1)
  %final = trunc i15 %pown2 to i8
  store i8 %final, ptr %out, align 1
  ret void
}
declare dso_local spir_func i50 @_Z30__spirv_ArbitraryFloatLogINTELILi50ELi50EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i50, i32, i32, i32, i32, i32) 
declare dso_local spir_func i38 @_Z31__spirv_ArbitraryFloatLog2INTELILi38ELi38EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i50, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i10 @_Z32__spirv_ArbitraryFloatLog10INTELILi8ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i38 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func i49 @_Z32__spirv_ArbitraryFloatLog1pINTELILi48ELi49EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i50, i32, i32, i32, i32, i32) 
declare dso_local spir_func i42 @_Z30__spirv_ArbitraryFloatExpINTELILi42ELi42EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i42, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i5 @_Z31__spirv_ArbitraryFloatExp2INTELILi3ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i10 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i25 @_Z32__spirv_ArbitraryFloatExp10INTELILi25ELi25EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i42 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func i62 @_Z32__spirv_ArbitraryFloatExpm1INTELILi64ELi62EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i64, i32, i32, i32, i32, i32) 
declare dso_local spir_func i34 @_Z30__spirv_ArbitraryFloatSinINTELILi30ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i25 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i4 @_Z30__spirv_ArbitraryFloatCosINTELILi4ELi4EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i5 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func i62 @_Z33__spirv_ArbitraryFloatSinCosINTELILi27ELi31EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(i34 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i13 @_Z32__spirv_ArbitraryFloatSinPiINTELILi10ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i10 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func i59 @_Z32__spirv_ArbitraryFloatCosPiINTELILi59ELi59EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i62, i32, i32, i32, i32, i32) 
declare dso_local spir_func i64 @_Z35__spirv_ArbitraryFloatSinCosPiINTELILi30ELi32EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEiiiiii(i34 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i11 @_Z31__spirv_ArbitraryFloatASinINTELILi7ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i4 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func i35 @_Z33__spirv_ArbitraryFloatASinPiINTELILi35ELi35EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i49, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i14 @_Z31__spirv_ArbitraryFloatACosINTELILi14ELi14EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i13 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i8 @_Z33__spirv_ArbitraryFloatACosPiINTELILi8ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i38 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func i44 @_Z31__spirv_ArbitraryFloatATanINTELILi44ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i59, i32, i32, i32, i32, i32) 
declare dso_local spir_func i34 @_Z33__spirv_ArbitraryFloatATanPiINTELILi40ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEiiiiii(i25, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i27 @_Z32__spirv_ArbitraryFloatATan2INTELILi24ELi25ELi27EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i25 signext, i32, i25 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i21 @_Z30__spirv_ArbitraryFloatPowINTELILi17ELi19ELi21EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i17 signext, i32, i19 signext, i32, i32, i32, i32, i32) 
declare dso_local spir_func i56 @_Z31__spirv_ArbitraryFloatPowRINTELILi54ELi55ELi56EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiiii(i59, i32, i55, i32, i32, i32, i32, i32) 
declare dso_local spir_func signext i15 @_Z31__spirv_ArbitraryFloatPowNINTELILi12ELi10ELi15EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiii(i14 signext, i32, i10 signext, i1 zeroext, i32, i32, i32, i32) 
declare dso_local spir_func signext i15 @_Z31__spirv_ArbitraryFloatPowNINTELILi64ELi10ELi15EEU7_ExtIntIXT1_EEiU7_ExtIntIXT_EEiiU7_ExtIntIXT0_EEiiiii(i64, i32, i10 signext, i1 zeroext, i32, i32, i32, i32) 
