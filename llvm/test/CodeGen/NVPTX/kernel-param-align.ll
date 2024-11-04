; RUN: llc < %s -march=nvptx64 -mcpu=sm_60 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_60 | %ptxas-verify %}

%struct.Large = type { [16 x double] }

; CHECK-LABEL: .entry func_align(
; CHECK: .param .u64 .ptr .global .align 16 func_align_param_0
; CHECK: .param .u64 .ptr .global .align 16 func_align_param_1
; CHECK: .param .u64 .ptr .global .align 16 func_align_param_2
; CHECK: .param .u64 .ptr .shared .align 16 func_align_param_3
; CHECK: .param .u64 .ptr .const  .align 16 func_align_param_4
define void @func_align(ptr nocapture readonly align 16 %input,
                        ptr nocapture align 16 %out,
                        ptr addrspace(1) align 16 %global,
                        ptr addrspace(3) align 16 %shared,
                        ptr addrspace(4) align 16 %const) {
entry:
  %0 = addrspacecast ptr %out to ptr addrspace(1)
  %1 = addrspacecast ptr %input to ptr addrspace(1)
  %getElem = getelementptr inbounds %struct.Large, ptr addrspace(1) %1, i64 0, i32 0, i64 5
  %tmp2 = load i32, ptr addrspace(1) %getElem, align 8
  store i32 %tmp2, ptr addrspace(1) %0, align 4
  ret void
}

; CHECK-LABEL: .entry func_noalign(
; CHECK: .param .u64 .ptr .global func_noalign_param_0
; CHECK: .param .u64 .ptr .global func_noalign_param_1
; CHECK: .param .u64 .ptr .global func_noalign_param_2
; CHECK: .param .u64 .ptr .shared func_noalign_param_3
; CHECK: .param .u64 .ptr .const func_noalign_param_4
define void @func_noalign(ptr nocapture readonly %input,
                          ptr nocapture %out,
                          ptr addrspace(1) %global,
                          ptr addrspace(3) %shared,
                          ptr addrspace(4) %const) {
entry:
  %0 = addrspacecast ptr %out to ptr addrspace(1)
  %1 = addrspacecast ptr %input to ptr addrspace(1)
  %getElem = getelementptr inbounds %struct.Large, ptr addrspace(1) %1, i64 0, i32 0, i64 5
  %tmp2 = load i32, ptr addrspace(1) %getElem, align 8
  store i32 %tmp2, ptr addrspace(1) %0, align 4
  ret void
}

!nvvm.annotations = !{!0, !1}
!0 = !{ptr @func_align, !"kernel", i32 1}
!1 = !{ptr @func_noalign, !"kernel", i32 1}
