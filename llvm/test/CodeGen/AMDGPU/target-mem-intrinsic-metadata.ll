; RUN: llc -global-isel=0 -march=amdgcn -mcpu=bonaire -stop-before=machine-scheduler < %s | FileCheck -enable-var-scope -check-prefixes=MIR %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=bonaire -stop-before=machine-scheduler < %s | FileCheck -enable-var-scope -check-prefixes=MIR %s

; Make sure !noalias metadata is passed through from target intrinsics

; MIR-LABEL: name: ds_append_noalias
; MIR: DS_APPEND {{.*}} :: (load store (s32) on %{{.*}}, !noalias !{{[0-9]+}}, addrspace 3)
define amdgpu_kernel void @ds_append_noalias() {
  %lds = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(1)* null
  %val = call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %lds, i1 false), !noalias !0
  store i32 %val, i32 addrspace(1)* null, align 4
  ret void
}

declare i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* nocapture, i1 immarg) #0

attributes #0 = { argmemonly convergent nounwind willreturn }

!0 = !{!1}
!1 = distinct !{!1, !2}
!2 = distinct !{!2}
