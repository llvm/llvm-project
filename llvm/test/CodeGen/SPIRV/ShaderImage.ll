; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val %}

; CHECK: [[Float:%[0-9]+]] = OpTypeFloat 32
; CHECK: [[Void:%[0-9]+]] = OpTypeVoid
; CHECK: [[ImageType:%[0-9]+]] = OpTypeImage [[Float]] Buffer 2 0 0 1 R32i {{$}}
; CHECK: [[ImageFuncType:%[0-9]+]] = OpTypeFunction [[Void]] [[ImageType]]
; CHECK: [[SampledImageType:%[0-9]+]] = OpTypeSampledImage [[ImageType]]
; CHECK: [[SampledImageFuncType:%[0-9]+]] = OpTypeFunction [[Void]] [[SampledImageType]]

; CHECK: {{%[0-9]+}} = OpFunction [[Void]] DontInline [[ImageFuncType]]    ; -- Begin function ImageWithNoAccessQualifier
define void @ImageWithNoAccessQualifier(target("spirv.Image", float, 5, 2, 0, 0, 1, 24) %img) #0 {
  ret void
}

; CHECK: {{%[0-9]+}} = OpFunction [[Void]] DontInline [[SampledImageFuncType]]    ; -- Begin function SampledImageWithNoAccessQualifier
define void @SampledImageWithNoAccessQualifier(target("spirv.SampledImage", float, 5, 2, 0, 0, 1, 24) %img) #0 {
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
