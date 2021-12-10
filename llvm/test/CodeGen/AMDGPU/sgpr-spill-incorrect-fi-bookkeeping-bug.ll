; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s

; This tests for a bug that caused a crash in SIRegisterInfo::spillSGPR()
; which was due to incorrect book-keeping of removed dead frame indices.

; CHECK-LABEL: {{^}}kernel0:
define amdgpu_kernel void @kernel0(i32 addrspace(1)* %out, i32 %in) #1 {
  call void asm sideeffect "", "~{v[0:7]}" () #0
  call void asm sideeffect "", "~{v[8:15]}" () #0
  call void asm sideeffect "", "~{v[16:19]}"() #0
  call void asm sideeffect "", "~{v[20:21]}"() #0
  call void asm sideeffect "", "~{v22}"() #0
  %val0 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0
  %val1 = call <4 x i32> asm sideeffect "; def $0", "=s" () #0
  %val2 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %val3 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %val4 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0
  %val5 = call <4 x i32> asm sideeffect "; def $0", "=s" () #0
  %val6 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %val7 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %val8 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0
  %val9 = call <4 x i32> asm sideeffect "; def $0", "=s" () #0
  %val10 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %val11 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %val12 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0
  %val13 = call <4 x i32> asm sideeffect "; def $0", "=s" () #0
  %val14 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %val15 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %val16 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0
  %val17 = call <4 x i32> asm sideeffect "; def $0", "=s" () #0
  %val18 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %val19 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val0) #0
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val1) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val2) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val3) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val4) #0
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val5) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val6) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val7) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val8) #0
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val9) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val10) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val11) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val12) #0
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val13) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val14) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val15) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %val16) #0
  call void asm sideeffect "; use $0", "s"(<4 x i32> %val17) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %val18) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %val19) #0
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-waves-per-eu"="10,10" }
