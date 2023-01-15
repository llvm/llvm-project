; RUN: not llc < %s -verify-machineinstrs -mtriple=armv7-none-linux-gnu -mattr=+neon 2>&1 | FileCheck %s

%struct.float4 = type { float, float, float, float }

; CHECK: error: Don't know how to handle indirect register inputs yet for constraint 'w'
define float @inline_func(float %f1, float %f2) #0 {
  %c1 = alloca %struct.float4, align 4
  %c2 = alloca %struct.float4, align 4
  %c3 = alloca %struct.float4, align 4
  call void asm sideeffect "vmul.f32 ${2:q}, ${0:q}, ${1:q}", "=*r,=*r,*w"(ptr elementtype(%struct.float4) %c1, ptr elementtype(%struct.float4) %c2, ptr elementtype(%struct.float4) %c3) #1, !srcloc !1
  %1 = load float, ptr %c3, align 4
  ret float %1
}

!1 = !{i32 271, i32 305}
