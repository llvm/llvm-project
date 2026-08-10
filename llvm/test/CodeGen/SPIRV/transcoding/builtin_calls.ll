; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpDecorate %[[#Id0:]] BuiltIn GlobalLinearId
; CHECK-SPIRV-DAG: OpDecorate %[[#Id1:]] BuiltIn GlobalInvocationId
; CHECK-SPIRV-DAG: OpDecorate %[[#Id2:]] BuiltIn LocalInvocationIndex
; CHECK-SPIRV-DAG: OpDecorate %[[#Id3:]] BuiltIn WorkDim
; CHECK-SPIRV-DAG: OpDecorate %[[#Id4:]] BuiltIn SubgroupSize
; CHECK-SPIRV-DAG: OpDecorate %[[#Id5:]] BuiltIn SubgroupMaxSize
; CHECK-SPIRV-DAG: OpDecorate %[[#Id6:]] BuiltIn NumSubgroups
; CHECK-SPIRV-DAG: OpDecorate %[[#Id7:]] BuiltIn NumEnqueuedSubgroups
; CHECK-SPIRV-DAG: OpDecorate %[[#Id8:]] BuiltIn SubgroupId
; CHECK-SPIRV-DAG: OpDecorate %[[#Id9:]] BuiltIn SubgroupLocalInvocationId
; CHECK-SPIRV-DAG: OpDecorate %[[#Id10:]] BuiltIn SubgroupEqMask
; CHECK-SPIRV-DAG: OpDecorate %[[#Id11:]] BuiltIn SubgroupGeMask
; CHECK-SPIRV-DAG: OpDecorate %[[#Id12:]] BuiltIn SubgroupGtMask
; CHECK-SPIRV-DAG: OpDecorate %[[#Id13:]] BuiltIn SubgroupLeMask
; CHECK-SPIRV-DAG: OpDecorate %[[#Id14:]] BuiltIn SubgroupLtMask
; CHECK-SPIRV-DAG: OpDecorate %[[#Id15:]] BuiltIn LocalInvocationId
; CHECK-SPIRV-DAG: OpDecorate %[[#Id16:]] BuiltIn WorkgroupSize
; CHECK-SPIRV-DAG: OpDecorate %[[#Id17:]] BuiltIn GlobalSize
; CHECK-SPIRV-DAG: OpDecorate %[[#Id18:]] BuiltIn WorkgroupId
; CHECK-SPIRV-DAG: OpDecorate %[[#Id19:]] BuiltIn EnqueuedWorkgroupSize
; CHECK-SPIRV-DAG: OpDecorate %[[#Id20:]] BuiltIn NumWorkgroups
; CHECK-SPIRV-DAG: OpDecorate %[[#Id21:]] BuiltIn GlobalOffset

; CHECK-SPIRV:     %[[#Id0:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id1:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id2:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id3:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id4:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id5:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id6:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id7:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id8:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id9:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id10:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id11:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id12:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id13:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id14:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id15:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id16:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id17:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id18:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id19:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id20:]] = OpVariable %[[#]] Input
; CHECK-SPIRV:     %[[#Id21:]] = OpVariable %[[#]] Input

define spir_kernel void @f() {
entry:
  %0 = call spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()
  %1 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1)
  %2 = call spir_func i64 @_Z35__spirv_BuiltInLocalInvocationIndexv()
  %3 = call spir_func i32 @_Z22__spirv_BuiltInWorkDimv()
  %4 = call spir_func i32 @_Z27__spirv_BuiltInSubgroupSizev()
  %5 = call spir_func i32 @_Z30__spirv_BuiltInSubgroupMaxSizev()
  %6 = call spir_func i32 @_Z27__spirv_BuiltInNumSubgroupsv()
  %7 = call spir_func i32 @_Z35__spirv_BuiltInNumEnqueuedSubgroupsv()
  %8 = call spir_func i32 @_Z25__spirv_BuiltInSubgroupIdv()
  %9 = call spir_func i32 @_Z40__spirv_BuiltInSubgroupLocalInvocationIdv()
  %10 = call spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupEqMaskv()
  %11 = call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupEqMaskKHRv()
  %12 = call spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupGeMaskv()
  %13 = call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupGeMaskKHRv()
  %14 = call spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupGtMaskv()
  %15 = call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupGtMaskKHRv()
  %16 = call spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupLeMaskv()
  %17 = call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupLeMaskKHRv()
  %18 = call spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupLtMaskv()
  %19 = call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupLtMaskKHRv()
  %20 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0)
  %21 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0)
  %22 = call spir_func i64 @_Z25__spirv_BuiltInGlobalSizei(i32 0)
  %23 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0)
  %24 = call spir_func i64 @_Z36__spirv_BuiltInEnqueuedWorkgroupSizei(i32 0)
  %25 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 0)
  %26 = call spir_func i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 0)

  ret void
}

declare spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32)
declare spir_func i64 @_Z35__spirv_BuiltInLocalInvocationIndexv()
declare spir_func i32 @_Z22__spirv_BuiltInWorkDimv()
declare spir_func i32 @_Z27__spirv_BuiltInSubgroupSizev()
declare spir_func i32 @_Z30__spirv_BuiltInSubgroupMaxSizev()
declare spir_func i32 @_Z27__spirv_BuiltInNumSubgroupsv()
declare spir_func i32 @_Z35__spirv_BuiltInNumEnqueuedSubgroupsv()
declare spir_func i32 @_Z25__spirv_BuiltInSubgroupIdv()
declare spir_func i32 @_Z40__spirv_BuiltInSubgroupLocalInvocationIdv()
declare spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupEqMaskv()
declare spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupEqMaskKHRv()
declare spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupGeMaskv()
declare spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupGeMaskKHRv()
declare spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupGtMaskv()
declare spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupGtMaskKHRv()
declare spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupLeMaskv()
declare spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupLeMaskKHRv()
declare spir_func <4 x i32> @_Z29__spirv_BuiltInSubgroupLtMaskv()
declare spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupLtMaskKHRv()
declare spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32)
declare spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32)
declare spir_func i64 @_Z25__spirv_BuiltInGlobalSizei(i32)
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32)
declare spir_func i64 @_Z36__spirv_BuiltInEnqueuedWorkgroupSizei(i32)
declare spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32)
declare spir_func i64 @_Z27__spirv_BuiltInGlobalOffseti(i32)
