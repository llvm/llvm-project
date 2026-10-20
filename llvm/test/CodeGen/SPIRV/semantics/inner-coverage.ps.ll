; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-pixel --spirv-ext=+SPV_EXT_fragment_fully_covered %s -o - | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv1.6-vulkan1.3-pixel %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-pixel --spirv-ext=+SPV_EXT_fragment_fully_covered %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; The FullyCoveredEXT built-in requires the FragmentFullyCoveredEXT capability,
; which is enabled by the SPV_EXT_fragment_fully_covered extension.
; ERROR: LLVM ERROR: FullyCoveredEXT requires the following SPIR-V extension: SPV_EXT_fragment_fully_covered

; CHECK-DAG: OpCapability FragmentFullyCoveredEXT
; CHECK-DAG: OpExtension "SPV_EXT_fragment_fully_covered"
; CHECK-DAG: OpDecorate %[[#INPUT:]] BuiltIn FullyCoveredEXT

; CHECK-DAG:   %[[#bool:]] = OpTypeBool
; CHECK-DAG:  %[[#ptr_i:]] = OpTypePointer Input %[[#bool]]
; CHECK-DAG:  %[[#INPUT]] = OpVariable %[[#ptr_i]] Input

@SV_InnerCoverage = external hidden thread_local addrspace(7) externally_initialized constant i1, !spirv.Decorations !0
@A0 = external hidden thread_local addrspace(8) global i32, !spirv.Decorations !2

define void @main() #0 {
entry:
  %0 = load i1, ptr addrspace(7) @SV_InnerCoverage, align 1
  %1 = zext i1 %0 to i32
  store i32 %1, ptr addrspace(8) @A0, align 4
  ret void

; CHECK: %[[#TMP:]] = OpLoad %[[#bool]] %[[#INPUT]]
; CHECK:              OpSelect %[[#]] %[[#TMP]]
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1}
!1 = !{i32 11, i32 5264}
!2 = !{!3}
!3 = !{i32 30, i32 0}
