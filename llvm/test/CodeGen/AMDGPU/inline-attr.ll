; RUN: opt -mtriple=amdgcn--amdhsa -S -O3 -enable-unsafe-fp-math %s  | FileCheck -check-prefix=GCN -check-prefix=UNSAFE %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -O3 -enable-no-nans-fp-math %s | FileCheck -check-prefix=GCN -check-prefix=NONANS %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -O3 -enable-no-infs-fp-math %s | FileCheck -check-prefix=GCN -check-prefix=NOINFS %s

; GCN: define float @foo(float %x) local_unnamed_addr #0 {
; GCN: define amdgpu_kernel void @caller(ptr addrspace(1) nocapture %p) local_unnamed_addr #1 {
; GCN: %mul.i = fmul float %load, 1.500000e+01

; UNSAFE: attributes #0 = { nounwind "amdgpu-waves-per-eu"="4,10" "uniform-work-group-size"="false" "unsafe-fp-math"="true" }
; UNSAFE: attributes #1 = { nounwind "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "uniform-work-group-size"="false" "unsafe-fp-math"="true" }

; NOINFS: attributes #0 = { nounwind "amdgpu-waves-per-eu"="4,10" "no-infs-fp-math"="true" "uniform-work-group-size"="false" }
; NOINFS: attributes #1 = { nounwind "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="false" "uniform-work-group-size"="false" "unsafe-fp-math"="false" }

; NONANS: attributes #0 = { nounwind "amdgpu-waves-per-eu"="4,10" "no-nans-fp-math"="true" "uniform-work-group-size"="false" }
; NONANS: attributes #1 = { nounwind "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="true" "uniform-work-group-size"="false" "unsafe-fp-math"="false" }

declare void @extern() #0

define float @foo(float %x) #0 {
entry:
  call void @extern()
  %mul = fmul float %x, 1.500000e+01
  ret float %mul
}

define amdgpu_kernel void @caller(ptr addrspace(1) %p) #1 {
entry:
  %load = load float, ptr addrspace(1) %p, align 4
  %call = call fast float @foo(float %load)
  store float %call, ptr addrspace(1) %p, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "less-precise-fpmad"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
