; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=CHECK,CORE
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: llc -O0 -mtriple=spirv1.5-unknown-vulkan1.2-compute --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.5-unknown-vulkan1.2-compute --spirv-ext=+SPV_KHR_non_semantic_info %s -o - -filetype=obj | spirv-val --target-env vulkan1.2 %}
; Without the extension on SPIR-V < 1.6 the set is unavailable, so lowering
; falls back to OpNop.
; RUN: llc -O0 -mtriple=spirv1.5-unknown-vulkan1.2-compute %s -o - | FileCheck %s --check-prefix=NOEXT
; Two consecutive debugtraps must both survive -O2 and lower to two DebugBreak.
; RUN: llc -O2 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefix=O2

; CORE-NOT:  OpExtension "SPV_KHR_non_semantic_info"
; EXT:       OpExtension "SPV_KHR_non_semantic_info"
; CHECK:     %[[#Set:]] = OpExtInstImport "NonSemantic.DebugBreak"
; CHECK:     OpFunction
; CHECK:     OpExtInst %[[#]] %[[#Set]] DebugBreak
; CHECK:     OpExtInst %[[#]] %[[#Set]] DebugBreak
; CHECK:     OpReturn

; NOEXT-NOT: OpExtInstImport "NonSemantic.DebugBreak"
; NOEXT:     OpNop
; NOEXT:     OpNop
; NOEXT:     OpReturn

; O2:        %[[#Set:]] = OpExtInstImport "NonSemantic.DebugBreak"
; O2:        OpExtInst %[[#]] %[[#Set]] DebugBreak
; O2-NEXT:   OpExtInst %[[#]] %[[#Set]] DebugBreak

define void @main() #0 {
entry:
  call void @llvm.debugtrap()
  call void @llvm.debugtrap()
  ret void
}

declare void @llvm.debugtrap()

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
