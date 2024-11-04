; RUN: llc < %s -march=nvptx64 -mcpu=sm_72 2>&1 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_72 | %ptxas-verify %}

%struct.Large = type { [16 x double] }

; CHECK-LABEL: .entry func_align(
; CHECK: .param .u64 .ptr .global .align 16 func_align_param_0,
; CHECK: .param .u64 .ptr .global func_align_param_1,
; CHECK: .param .u64 .ptr .global func_align_param_2
define void @func_align(ptr nocapture readonly align 16 %input, ptr nocapture %out, ptr addrspace(3) %n) {
entry:
  %0 = addrspacecast ptr %out to ptr addrspace(1)
  %1 = addrspacecast ptr %input to ptr addrspace(1)
  %getElem = getelementptr inbounds %struct.Large, ptr addrspace(1) %1, i64 0, i32 0, i64 5
  %tmp2 = load i32, ptr addrspace(1) %getElem, align 8
  store i32 %tmp2, ptr addrspace(1) %0, align 4
  ret void
}

; CHECK-LABEL: .entry func(
; CHECK: .param .u64 .ptr .global func_param_0,
; CHECK: .param .u64 .ptr .global func_param_1,
; CHECK: .param .u32 func_param_2
define void @func(ptr nocapture readonly %input, ptr nocapture %out, i32 %n) {
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
!1 = !{ptr @func, !"kernel", i32 1}