; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -S -o - %s | FileCheck %s

; This is NOT OK to vectorize, as either load may alias either store.

; CHECK: load double
; CHECK: store double 0.000000e+00, ptr addrspace(1) %a,
; CHECK: load double
; CHECK: store double 0.000000e+00, ptr addrspace(1) %a.idx.1
define amdgpu_kernel void @interleave(ptr addrspace(1) nocapture %a, ptr addrspace(1) nocapture %b, ptr addrspace(1) nocapture readonly %c) #0 {
entry:
  %a.idx.1 = getelementptr inbounds double, ptr addrspace(1) %a, i64 1
  %c.idx.1 = getelementptr inbounds double, ptr addrspace(1) %c, i64 1

  %ld.c = load double, ptr addrspace(1) %c, align 8 ; may alias store to %a
  store double 0.0, ptr addrspace(1) %a, align 8

  %ld.c.idx.1 = load double, ptr addrspace(1) %c.idx.1, align 8 ; may alias store to %a
  store double 0.0, ptr addrspace(1) %a.idx.1, align 8

  %add = fadd double %ld.c, %ld.c.idx.1
  store double %add, ptr addrspace(1) %b

  ret void
}

attributes #0 = { nounwind }
