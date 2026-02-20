; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

%A = type {
  i32,
  i32
}

%B = type {
  %A,
  i32,
  %A
}

; The pointer types and the struct types could be removed,
; but we fail to do so due to issues with `validatePtrTypes` function.
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#FnTy:]] = OpTypeFunction %[[#Void]]
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int32Ptr:]] = OpTypePointer Function %[[#Int32]]
; CHECK-DAG: %[[#A:]] = OpTypeStruct %[[#Int32]] %[[#Int32]]
; CHECK-DAG: %[[#APtr:]] = OpTypePointer Function %[[#A]]
; CHECK-DAG: %[[#B:]] = OpTypeStruct %[[#A]] %[[#Int32]] %[[#A]]
; CHECK-DAG: %[[#BPtr:]] = OpTypePointer Function %[[#B]]

; Make sure the GEPs and the function scope variable are removed.
; CHECK: OpFunction %[[#Void]] None %[[#FnTy]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define void @main() #1 {
entry:
  %0 = alloca %B, align 4
  %1 = getelementptr %B, ptr %0, i32 0, i32 2
  %2 = getelementptr %A, ptr %1, i32 0, i32 1
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
