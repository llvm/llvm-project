; RUN: opt -passes=slp-vectorizer -mtriple=amdgpu9.42-amd-amdhsa -pass-remarks=slp-vectorizer -disable-output < %s 2>&1 | FileCheck %s

; The second fmul of each fsub/fadd does not fuse, so packing those pays off.
; CHECK: Stores SLP vectorized with cost -2
define void @cmul_store(ptr addrspace(1) %out, float %xr, float %xi, float %wr, float %wi) {
  %xr.wr = fmul contract float %xr, %wr
  %xi.wi = fmul contract float %xi, %wi
  %re = fsub contract float %xr.wr, %xi.wi
  %xi.wr = fmul contract float %xi, %wr
  %xr.wi = fmul contract float %xr, %wi
  %im = fadd contract float %xi.wr, %xr.wi
  store float %re, ptr addrspace(1) %out, align 8
  %out.im = getelementptr inbounds float, ptr addrspace(1) %out, i64 1
  store float %im, ptr addrspace(1) %out.im, align 4
  ret void
}
