; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bindless_images %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: LLVM ERROR: Parameter value must be a 32-bit scalar in case of Physical32 addressing model or a 64-bit scalar in case of Physical64 addressing model

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @foo(i32 %in) {
  %img = call spir_func target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELi(i32 %in)
  %samp = call spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELl(i64 42)
  %sampImage = call spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z40__spirv_ConvertHandleToSampledImageINTELl(i64 43)
  ret void
}

declare spir_func target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELi(i32)

declare spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELl(i64)

declare spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z40__spirv_ConvertHandleToSampledImageINTELl(i64)
