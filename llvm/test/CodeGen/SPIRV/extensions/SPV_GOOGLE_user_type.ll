; RUN: split-file %s %t
; RUN: llc -O0 -mtriple=spirv32v1.0-unknown-unknown --spirv-ext=+SPV_GOOGLE_user_type %t/opencl.ll -o - | FileCheck %s --check-prefix=OPENCL
; RUN: llc -O0 -mtriple=spirv32v1.4-unknown-unknown --spirv-ext=+SPV_GOOGLE_user_type %t/opencl.ll -o - | FileCheck %s --check-prefix=OPENCL
; RUN: llc -O0 -mtriple=spirv64v1.4-unknown-unknown --spirv-ext=+SPV_GOOGLE_user_type %t/opencl.ll -o - | FileCheck %s --check-prefix=OPENCL
; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_GOOGLE_user_type %t/vulkan.ll -o - | FileCheck %s --check-prefix=VULKAN
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.4-unknown-unknown --spirv-ext=+SPV_GOOGLE_user_type %t/opencl.ll -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.4-unknown-unknown --spirv-ext=+SPV_GOOGLE_user_type %t/opencl.ll -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_GOOGLE_user_type %t/vulkan.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: not --crash llc -O0 -mtriple=spirv64v1.4-unknown-unknown %t/opencl.ll -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not --crash llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %t/vulkan.ll -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -mtriple=spirv64v1.4-unknown-unknown --spirv-ext=+SPV_GOOGLE_user_type %t/unused.ll -o - | FileCheck %s --check-prefix=UNUSED

; UserTypeGOOGLE requires OpDecorateString and the extension declaration.
; Distinct type strings on the same variable must survive deduplication,
; while identical decorations should be emitted only once.
; OPENCL: OpExtension "SPV_GOOGLE_user_type"
; OPENCL: OpName %[[#Buffer:]] "buffer"
; OPENCL-NOT: UserTypeGOOGLE
; OPENCL-COUNT-1: OpDecorateString %[[#Buffer]] UserTypeGOOGLE "buffer:<uint>"
; OPENCL-NOT: UserTypeGOOGLE
; OPENCL-COUNT-1: OpDecorateString %[[#Buffer]] UserTypeGOOGLE "custom_buffer"
; OPENCL-NOT: UserTypeGOOGLE
; OPENCL: %[[#Buffer]] = OpVariable

; VULKAN: OpExtension "SPV_GOOGLE_user_type"
; VULKAN: OpName %[[#Local:]] "local"
; VULKAN-NOT: UserTypeGOOGLE
; VULKAN-COUNT-1: OpDecorateString %[[#Local]] UserTypeGOOGLE "local_value"
; VULKAN-NOT: UserTypeGOOGLE
; VULKAN: %[[#Local]] = OpVariable

; ERROR: LLVM ERROR: Adding SPIR-V requirements this target can't satisfy.
; UNUSED-NOT: OpExtension "SPV_GOOGLE_user_type"

;--- opencl.ll
@buffer = addrspace(1) global i32 0, !spirv.Decorations !0

define spir_kernel void @kernel() {
  %value = load i32, ptr addrspace(1) @buffer
  ret void
}

!0 = !{!1, !2, !1}
!1 = !{i32 5636, !"buffer:<uint>"}
!2 = !{i32 5636, !"custom_buffer"}

;--- vulkan.ll
define void @main() #0 {
  %local = alloca i32, align 4, !spirv.Decorations !0
  store volatile i32 1, ptr %local
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!0 = !{!1, !1}
!1 = !{i32 5636, !"local_value"}

;--- unused.ll
define spir_kernel void @kernel() {
  ret void
}
