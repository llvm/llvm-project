; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability ImageBasic
; CHECK-NOT: OpCapability ImageReadWrite
; CHECK: OpCapability ImageBuffer
; CHECK-NOT: OpCapability ImageBasic
; CHECK-NOT: OpCapability ImageReadWrite

; CHECK-DAG: [[Float:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[Void:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[ImageType:%[0-9]+]] = OpTypeImage [[Float]] Buffer 2 0 0 2 R32i {{$}}
; CHECK-DAG: [[ImageFuncType:%[0-9]+]] = OpTypeFunction [[Void]] [[ImageType]]

; CHECK: {{%[0-9]+}} = OpFunction [[Void]] DontInline [[ImageFuncType]]
define void @ImageWithNoAccessQualifier(target("spirv.Image", float, 5, 2, 0, 0, 2, 24) %img) #0 {
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
