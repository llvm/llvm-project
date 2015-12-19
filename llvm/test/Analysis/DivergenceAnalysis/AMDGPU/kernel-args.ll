; RUN: opt %s -mtriple amdgcn-- -analyze -divergence | FileCheck %s

; CHECK: DIVERGENT:
; CHECK-NOT: %arg0
; CHECK-NOT: %arg1
; CHECK-NOT; %arg2
; CHECK: <2 x i32> %arg3
; CHECK: DIVERGENT:  <3 x i32> %arg4
; CHECK: DIVERGENT:  float %arg5
; CHECK: DIVERGENT:  i32 %arg6

define void @main([4 x <16 x i8>] addrspace(2)* byval %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}

attributes #0 = { "ShaderType"="0" }
