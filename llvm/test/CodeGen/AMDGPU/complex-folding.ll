;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: {{^}}main:
; CHECK-NOT: MOV
define amdgpu_ps void @main(<4 x float> inreg %reg0) {
entry:
  %0 = extractelement <4 x float> %reg0, i32 0
  %1 = call float @fabsf(float %0)
  %2 = fptoui float %1 to i32
  %3 = bitcast i32 %2 to float
  %4 = insertelement <4 x float> undef, float %3, i32 0
  call void @llvm.r600.store.swizzle(<4 x float> %4, i32 0, i32 0)
  ret void
}

declare float @fabsf(float ) readnone
declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)
