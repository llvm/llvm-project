; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -passes=amdgpu-promote-alloca < %s | FileCheck %s

; The types of the users of the addrspacecast should not be changed.

; CHECK-LABEL: @invalid_bitcast_addrspace(
; CHECK: alloca
; CHECK: addrspacecast
; CHECK: load
; CHECK: bitcast
define amdgpu_kernel void @invalid_bitcast_addrspace() #0 {
entry:
  %data = alloca [1 x i32], addrspace(5)
  %tmp1 = addrspacecast ptr addrspace(5) %data to ptr
  %tmp3 = load <2 x i16>, ptr %tmp1, align 2
  %tmp4 = bitcast <2 x i16> %tmp3 to <2 x half>
  ret void
}

; A callee use is not promotable even if it has a nocapture attribute.
define void @nocapture_callee(ptr nocapture noundef writeonly %flat.observes.addrspace) #0 {
  %private.ptr = addrspacecast ptr %flat.observes.addrspace to ptr addrspace(5)
  store i32 1, ptr addrspace(5) %private.ptr, align 4
  ret void
}

; CHECK-LABEL: @kernel_call_nocapture(
; CHECK: alloca i32
; CHECK-NEXT: addrspacecast
; CHECK-NEXT: call
define amdgpu_kernel void @kernel_call_nocapture() #0 {
  %alloca = alloca i32, align 4, addrspace(5)
  %flat.alloca = addrspacecast ptr addrspace(5) %alloca to ptr
  call void @nocapture_callee(ptr noundef %flat.alloca)
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,256" }
