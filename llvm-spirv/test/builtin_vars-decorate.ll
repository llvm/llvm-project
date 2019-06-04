; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; ModuleID = 'test.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK: {{[0-9]+}} Name [[WD:[0-9]+]] "__spirv_BuiltInWorkDim"
; CHECK: {{[0-9]+}} Name [[GS:[0-9]+]] "__spirv_BuiltInGlobalSize"
; CHECK: {{[0-9]+}} Name [[GII:[0-9]+]] "__spirv_BuiltInGlobalInvocationId"
; CHECK: {{[0-9]+}} Name [[WS:[0-9]+]] "__spirv_BuiltInWorkgroupSize"
; CHECK: {{[0-9]+}} Name [[EWS:[0-9]+]] "__spirv_BuiltInEnqueuedWorkgroupSize"
; CHECK: {{[0-9]+}} Name [[LLI:[0-9]+]] "__spirv_BuiltInLocalInvocationId"
; CHECK: {{[0-9]+}} Name [[NW:[0-9]+]] "__spirv_BuiltInNumWorkgroups"
; CHECK: {{[0-9]+}} Name [[WI:[0-9]+]] "__spirv_BuiltInWorkgroupId"
; CHECK: {{[0-9]+}} Name [[GO:[0-9]+]] "__spirv_BuiltInGlobalOffset"
; CHECK: {{[0-9]+}} Name [[GLI:[0-9]+]] "__spirv_BuiltInGlobalLinearId"
; CHECK: {{[0-9]+}} Name [[LLII:[0-9]+]] "__spirv_BuiltInLocalInvocationIndex"
; CHECK: {{[0-9]+}} Name [[SS:[0-9]+]] "__spirv_BuiltInSubgroupSize"
; CHECK: {{[0-9]+}} Name [[SMS:[0-9]+]] "__spirv_BuiltInSubgroupMaxSize"
; CHECK: {{[0-9]+}} Name [[NS:[0-9]+]] "__spirv_BuiltInNumSubgroups"
; CHECK: {{[0-9]+}} Name [[NES:[0-9]+]] "__spirv_BuiltInNumEnqueuedSubgroups"
; CHECK: {{[0-9]+}} Name [[SI:[0-9]+]] "__spirv_BuiltInSubgroupId"
; CHECK: {{[0-9]+}} Name [[SLII:[0-9]+]] "__spirv_BuiltInSubgroupLocalInvocationId"

; CHECK: 4 Decorate [[NW]] BuiltIn 24
; CHECK: 4 Decorate [[WS]] BuiltIn 25
; CHECK: 4 Decorate [[WI]] BuiltIn 26
; CHECK: 4 Decorate [[LLI]] BuiltIn 27
; CHECK: 4 Decorate [[GII]] BuiltIn 28
; CHECK: 4 Decorate [[LLII]] BuiltIn 29
; CHECK: 4 Decorate [[WD]] BuiltIn 30
; CHECK: 4 Decorate [[GS]] BuiltIn 31
; CHECK: 4 Decorate [[EWS]] BuiltIn 32
; CHECK: 4 Decorate [[GO]] BuiltIn 33
; CHECK: 4 Decorate [[GLI]] BuiltIn 34
; CHECK: 4 Decorate [[SS]] BuiltIn 36
; CHECK: 4 Decorate [[SMS]] BuiltIn 37
; CHECK: 4 Decorate [[NS]] BuiltIn 38
; CHECK: 4 Decorate [[NES]] BuiltIn 39
; CHECK: 4 Decorate [[SI]] BuiltIn 40
; CHECK: 4 Decorate [[SLII]] BuiltIn 41
@__spirv_BuiltInWorkDim = external addrspace(1) global i32
@__spirv_BuiltInGlobalSize = external addrspace(1) global <3 x i32>
@__spirv_BuiltInGlobalInvocationId = external addrspace(1) global <3 x i32>
@__spirv_BuiltInWorkgroupSize = external addrspace(1) global <3 x i32>
@__spirv_BuiltInEnqueuedWorkgroupSize = external addrspace(1) global <3 x i32>
@__spirv_BuiltInLocalInvocationId = external addrspace(1) global <3 x i32>
@__spirv_BuiltInNumWorkgroups = external addrspace(1) global <3 x i32>
@__spirv_BuiltInWorkgroupId = external addrspace(1) global <3 x i32>
@__spirv_BuiltInGlobalOffset = external addrspace(1) global <3 x i32>
@__spirv_BuiltInGlobalLinearId = external addrspace(1) global i32
@__spirv_BuiltInLocalInvocationIndex = external addrspace(1) global i32
@__spirv_BuiltInSubgroupSize = external addrspace(1) global i32
@__spirv_BuiltInSubgroupMaxSize = external addrspace(1) global i32
@__spirv_BuiltInNumSubgroups = external addrspace(1) global i32
@__spirv_BuiltInNumEnqueuedSubgroups = external addrspace(1) global i32
@__spirv_BuiltInSubgroupId = external addrspace(1) global i32
@__spirv_BuiltInSubgroupLocalInvocationId = external addrspace(1) global i32

; Function Attrs: nounwind readnone
define spir_kernel void @_Z1wv() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  ret void
}

attributes #0 = { alwaysinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}
!llvm.ident = !{!9}
!spirv.Source = !{!10}
!spirv.String = !{!11}

!0 = !{}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 1}
!8 = !{}
!9 = !{!"clang version 3.6.1 "}
!10 = !{i32 3, i32 200000, !11}
!11 = !{!"test.cl"}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
