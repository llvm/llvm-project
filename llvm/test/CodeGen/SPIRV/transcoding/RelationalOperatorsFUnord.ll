; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:      %[[#bool:]] = OpTypeBool
; CHECK-SPIRV:      %[[#bool2:]] = OpTypeVector %[[#bool]] 2

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFUnordEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

@var = addrspace(1) global <2 x i1> zeroinitializer
define spir_kernel void @testFUnordEqual(<2 x float> %a, <2 x float> %b) {
entry:
  %0 = fcmp ueq <2 x float> %a, %b
  store <2 x i1> %0, <2 x i1> addrspace(1)* @var
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFUnordGreaterThan %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

define spir_kernel void @testFUnordGreaterThan(<2 x float> %a, <2 x float> %b) {
entry:
  %0 = fcmp ugt <2 x float> %a, %b
  store <2 x i1> %0, <2 x i1> addrspace(1)* @var
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFUnordGreaterThanEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

define spir_kernel void @testFUnordGreaterThanEqual(<2 x float> %a, <2 x float> %b) {
entry:
  %0 = fcmp uge <2 x float> %a, %b
  store <2 x i1> %0, <2 x i1> addrspace(1)* @var
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFUnordLessThan %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

define spir_kernel void @testFUnordLessThan(<2 x float> %a, <2 x float> %b) {
entry:
  %0 = fcmp ult <2 x float> %a, %b
  store <2 x i1> %0, <2 x i1> addrspace(1)* @var
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFUnordLessThanEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

define spir_kernel void @testFUnordLessThanEqual(<2 x float> %a, <2 x float> %b) {
entry:
  %0 = fcmp ule <2 x float> %a, %b
  store <2 x i1> %0, <2 x i1> addrspace(1)* @var
  ret void
}
