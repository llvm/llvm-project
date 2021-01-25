; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -inline --inline-threshold=1 < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=inline --inline-threshold=1 < %s | FileCheck %s

define hidden <16 x i32> @div_vecbonus(<16 x i32> %x, <16 x i32> %y) {
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
  ret <16 x i32> %div.12
}

; CHECK-LABEL: define amdgpu_kernel void @caller_vecbonus
; CHECK-NOT: udiv
; CHECK: tail call <16 x i32> @div_vecbonus
; CHECK: ret void
define amdgpu_kernel void @caller_vecbonus(<16 x i32> addrspace(1)* nocapture %x, <16 x i32> addrspace(1)* nocapture readonly %y) {
entry:
  %tmp = load <16 x i32>, <16 x i32> addrspace(1)* %x
  %tmp1 = load <16 x i32>, <16 x i32> addrspace(1)* %y
  %div.i = tail call <16 x i32> @div_vecbonus(<16 x i32> %tmp, <16 x i32> %tmp1)
  store <16 x i32> %div.i, <16 x i32> addrspace(1)* %x
  ret void
}
