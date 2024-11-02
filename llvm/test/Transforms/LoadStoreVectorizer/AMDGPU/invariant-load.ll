; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -S -o - %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; CHECK-LABEL: @interleave
; CHECK: load <2 x double>, <2 x double> addrspace(1)* %{{.}}, align 8{{$}}
; CHECK: store <2 x double> zeroinitializer
; CHECK: store double %add
define amdgpu_kernel void @interleave(double addrspace(1)* nocapture %a, double addrspace(1)* nocapture %b, double addrspace(1)* nocapture readonly %c) #0 {
entry:
  %a.idx.1 = getelementptr inbounds double, double addrspace(1)* %a, i64 1
  %c.idx.1 = getelementptr inbounds double, double addrspace(1)* %c, i64 1

  %ld.c = load double, double addrspace(1)* %c, align 8
  store double 0.0, double addrspace(1)* %a, align 8 ; Cannot alias invariant load

  %ld.c.idx.1 = load double, double addrspace(1)* %c.idx.1, align 8, !invariant.load !0
  store double 0.0, double addrspace(1)* %a.idx.1, align 8

  %add = fadd double %ld.c, %ld.c.idx.1
  store double %add, double addrspace(1)* %b

  ret void
}

attributes #0 = { nounwind }

!0 = !{}
