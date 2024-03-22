; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s

; This tests for a bug that caused a crash in SIRegisterInfo::spillSGPR()
; which was due to incorrect book-keeping of removed dead frame indices.

; CHECK-LABEL: {{^}}kernel0:
define amdgpu_kernel void @kernel0(ptr addrspace(1) %out, i32 %in) nounwind "amdgpu-waves-per-eu"="10,10" {
  call void asm sideeffect "", "~{v[0:7]}" () nounwind
  call void asm sideeffect "", "~{v[8:15]}" () nounwind
  call void asm sideeffect "", "~{v[16:19]}"() nounwind
  call void asm sideeffect "", "~{v[20:21]}"() nounwind
  call void asm sideeffect "", "~{v22}"() nounwind
  %val0 = call <2 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val1 = call <4 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val2 = call <8 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val3 = call <16 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val4 = call <2 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val5 = call <4 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val6 = call <8 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val7 = call <16 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val8 = call <2 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val9 = call <4 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val10 = call <8 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val11 = call <16 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val12 = call <2 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val13 = call <4 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val14 = call <8 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val15 = call <16 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val16 = call <2 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val17 = call <4 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val18 = call <8 x i32> asm sideeffect "; def $0", "=s" () nounwind
  %val19 = call <16 x i32> asm sideeffect "; def $0", "=s" () nounwind
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val0) nounwind
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val1) nounwind
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val2) nounwind
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val3) nounwind
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val4) nounwind
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val5) nounwind
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val6) nounwind
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val7) nounwind
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val8) nounwind
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val9) nounwind
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val10) nounwind
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val11) nounwind
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val12) nounwind
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val13) nounwind
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val14) nounwind
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val15) nounwind
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val16) nounwind
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val17) nounwind
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val18) nounwind
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val19) nounwind
  ret void
}
