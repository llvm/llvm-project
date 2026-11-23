; RUN: opt -S -mtriple=amdgpu12.50-amd-amdhsa -passes=amdgpu-create-profiling-clones < %s | FileCheck -check-prefixes=CHECK,CLONE %s
; RUN: opt -S -mtriple=amdgpu9.0a-amd-amdhsa -passes=amdgpu-create-profiling-clones < %s | FileCheck -check-prefixes=CHECK,NOCLONE %s

; A profiling clone is created for kernels which access LDS, on subtargets
; where the bank conflict model is implemented.

@lds = internal addrspace(3) global [64 x i32] poison, align 4

; A clone of a kernel in a comdat gets a comdat of its own.
; CHECK-DAG: $comdat_kernel = comdat any
; CLONE-DAG: $comdat_kernel.prof = comdat any
$comdat_kernel = comdat any

; CLONE-DAG: @uses_lds.prof.counters = external protected addrspace(1) global [0 x i32], align 4
; CLONE-DAG: @atomic_lds.prof.counters = external protected addrspace(1) global [0 x i32], align 4
; CLONE-DAG: @comdat_kernel.prof.counters = external protected addrspace(1) global [0 x i32], align 4
; NOCLONE-NOT: .prof

; CHECK-LABEL: define protected amdgpu_kernel void @uses_lds(
; CLONE-SAME: #[[ORIG:[0-9]+]]
define protected amdgpu_kernel void @uses_lds(i32 %i, i32 %v) {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %p, align 4
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @atomic_lds(
; CLONE-SAME: #[[ORIG]]
define amdgpu_kernel void @atomic_lds(i32 %i, i32 %v) {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
  %old = atomicrmw add ptr addrspace(3) %p, i32 %v seq_cst
  ret void
}

; Kernels which do not access LDS have nothing to measure.
; CHECK-LABEL: define amdgpu_kernel void @no_lds(
; CHECK-NOT: #
; CHECK-SAME: {
define amdgpu_kernel void @no_lds(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @opted_out(
; CHECK-SAME: #[[OPTOUT:[0-9]+]]
define amdgpu_kernel void @opted_out(i32 %i, i32 %v) "amdgpu-no-profiling-clone" {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %p, align 4
  ret void
}

; Frontends put every HIP kernel in a comdat, so that it may be defined in
; several translation units.
; CHECK-LABEL: define linkonce_odr amdgpu_kernel void @comdat_kernel(
; CLONE-SAME: #[[ORIG]] comdat {
define linkonce_odr amdgpu_kernel void @comdat_kernel(i32 %i, i32 %v) comdat {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %p, align 4
  ret void
}

; Without a comdat, the clone and the counters of a kernel which may be
; replaced at link time could not be deduplicated.
; CHECK-LABEL: define linkonce_odr amdgpu_kernel void @weak_kernel(
; CHECK-NOT: #
; CHECK-SAME: {
define linkonce_odr amdgpu_kernel void @weak_kernel(i32 %i, i32 %v) {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %p, align 4
  ret void
}

; The runtime could not find the clone of a local kernel.
; CHECK-LABEL: define internal amdgpu_kernel void @internal_kernel(
; CHECK-NOT: #
; CHECK-SAME: {
define internal amdgpu_kernel void @internal_kernel(i32 %i, i32 %v) {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %p, align 4
  ret void
}

; CHECK-LABEL: define void @not_a_kernel(
; CHECK-NOT: #
; CHECK-SAME: {
define void @not_a_kernel(i32 %i, i32 %v) {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %p, align 4
  ret void
}

; The clones have the same body, linkage and visibility as the kernels.
; CLONE-LABEL: define protected amdgpu_kernel void @uses_lds.prof(i32 %i, i32 %v)
; CLONE-SAME: #[[CLONE1:[0-9]+]]
; CLONE-NEXT:   %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
; CLONE-NEXT:   store i32 %v, ptr addrspace(3) %p, align 4
; CLONE-NEXT:   ret void

; CLONE-LABEL: define amdgpu_kernel void @atomic_lds.prof(i32 %i, i32 %v)
; CLONE-SAME: #[[CLONE2:[0-9]+]]
; CLONE-NEXT:   %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
; CLONE-NEXT:   %old = atomicrmw add ptr addrspace(3) %p, i32 %v seq_cst
; CLONE-NEXT:   ret void

; CLONE-LABEL: define linkonce_odr amdgpu_kernel void @comdat_kernel.prof(i32 %i, i32 %v)
; CLONE-SAME: #[[CLONE3:[0-9]+]] comdat {

; CLONE-NOT: define {{.*}}.prof

; CLONE-DAG: attributes #[[ORIG]] = { "amdgpu-lds-prof-original" }
; CHECK-DAG: attributes #[[OPTOUT]] = { "amdgpu-no-profiling-clone" }
; CLONE-DAG: attributes #[[CLONE1]] = { "amdgpu-lds-prof-clone"="uses_lds" }
; CLONE-DAG: attributes #[[CLONE2]] = { "amdgpu-lds-prof-clone"="atomic_lds" }
; CLONE-DAG: attributes #[[CLONE3]] = { "amdgpu-lds-prof-clone"="comdat_kernel" }
