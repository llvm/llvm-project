; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -inline --inline-threshold=1 --inlinehint-threshold=4 < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=inline --inline-threshold=1 --inlinehint-threshold=4 < %s | FileCheck %s

define hidden <16 x i32> @div_hint(<16 x i32> %x, <16 x i32> %y) #0 {
entry:
  %div.1 = udiv <16 x i32> %x, %y
  %div.2 = udiv <16 x i32> %div.1, %y
  %div.3 = udiv <16 x i32> %div.2, %y
  %div.4 = udiv <16 x i32> %div.3, %y
  %div.5 = udiv <16 x i32> %div.4, %y
  %div.6 = udiv <16 x i32> %div.5, %y
  %div.7 = udiv <16 x i32> %div.6, %y
  %div.8 = udiv <16 x i32> %div.7, %y
  %div.9 = udiv <16 x i32> %div.8, %y
  %div.10 = udiv <16 x i32> %div.9, %y
  %div.11 = udiv <16 x i32> %div.10, %y
  %div.12 = udiv <16 x i32> %div.11, %y
  %div.13 = udiv <16 x i32> %div.12, %y
  %div.14 = udiv <16 x i32> %div.13, %y
  %div.15 = udiv <16 x i32> %div.14, %y
  %div.16 = udiv <16 x i32> %div.15, %y
  %div.17 = udiv <16 x i32> %div.16, %y
  %div.18 = udiv <16 x i32> %div.17, %y
  %div.19 = udiv <16 x i32> %div.18, %y
  ret <16 x i32> %div.19
}

; CHECK-LABEL: define amdgpu_kernel void @caller_hint
; CHECK-NOT: call
; CHECK: udiv
; CHECK: ret void
define amdgpu_kernel void @caller_hint(<16 x i32> addrspace(1)* nocapture %x, <16 x i32> addrspace(1)* nocapture readonly %y) {
entry:
  %tmp = load <16 x i32>, <16 x i32> addrspace(1)* %x, align 4
  %tmp1 = load <16 x i32>, <16 x i32> addrspace(1)* %y, align 4
  %div.i = tail call <16 x i32> @div_hint(<16 x i32> %tmp, <16 x i32> %tmp1) #0
  store <16 x i32> %div.i, <16 x i32> addrspace(1)* %x, align 4
  ret void
}

define hidden <16 x i32> @div_nohint(<16 x i32> %x, <16 x i32> %y) {
entry:
  %div.1 = udiv <16 x i32> %x, %y
  %div.2 = udiv <16 x i32> %div.1, %y
  %div.3 = udiv <16 x i32> %div.2, %y
  %div.4 = udiv <16 x i32> %div.3, %y
  %div.5 = udiv <16 x i32> %div.4, %y
  %div.6 = udiv <16 x i32> %div.5, %y
  %div.7 = udiv <16 x i32> %div.6, %y
  %div.8 = udiv <16 x i32> %div.7, %y
  %div.9 = udiv <16 x i32> %div.8, %y
  %div.10 = udiv <16 x i32> %div.9, %y
  %div.11 = udiv <16 x i32> %div.10, %y
  %div.12 = udiv <16 x i32> %div.11, %y
  %div.13 = udiv <16 x i32> %div.12, %y
  %div.14 = udiv <16 x i32> %div.13, %y
  %div.15 = udiv <16 x i32> %div.14, %y
  %div.16 = udiv <16 x i32> %div.15, %y
  %div.17 = udiv <16 x i32> %div.16, %y
  %div.18 = udiv <16 x i32> %div.17, %y
  %div.19 = udiv <16 x i32> %div.18, %y
  ret <16 x i32> %div.19
}

; CHECK-LABEL: define amdgpu_kernel void @caller_nohint
; CHECK-NOT: udiv
; CHECK: tail call <16 x i32> @div_nohint
; CHECK: ret void
define amdgpu_kernel void @caller_nohint(<16 x i32> addrspace(1)* nocapture %x, <16 x i32> addrspace(1)* nocapture readonly %y) {
entry:
  %tmp = load <16 x i32>, <16 x i32> addrspace(1)* %x
  %tmp1 = load <16 x i32>, <16 x i32> addrspace(1)* %y
  %div.i = tail call <16 x i32> @div_nohint(<16 x i32> %tmp, <16 x i32> %tmp1)
  store <16 x i32> %div.i, <16 x i32> addrspace(1)* %x
  ret void
}

attributes #0 = { inlinehint }
