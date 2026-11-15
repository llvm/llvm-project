; RUN: opt -mtriple=amdgpu9.00 -S -structurizecfg -structurizecfg-skip-uniform-regions -verify-dom-info < %s | FileCheck %s
; RUN: opt -mtriple=amdgpu9.00 -S -passes='structurizecfg<skip-uniform-regions>,verify' < %s | FileCheck %s
; RUN: opt -mtriple=amdgpu9.00 -S -passes='structurizecfg,verify' < %s | FileCheck %s
; RUN: opt -mtriple=amdgpu9.00 -passes='print<regions>,structurizecfg<skip-uniform-regions>,print<regions>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=REGIONS
; RUN: llc -mtriple=r600 -stop-after=structurizecfg -verify-machineinstrs < %s | FileCheck %s --check-prefix=R600

; Only the top-level region contains both header => A and the outer latch A,
; because A exits through a uniform switch to two returns. The top-level
; region is never structurized, so flattening header => A into it would leave
; the divergent loop unstructured. Keep header => A and structurize it.
; R600 reaches this case without the GCN pipeline's exit-unifying passes.
;
; The region tree must also remain valid after structurization. Previously,
; flattening into the top-level region changed only the cached RegionInfo and
; reported no change, so the second dump lost header => A despite unchanged IR.
;
; REGIONS-LABEL: Region Tree for function: toplevel
; REGIONS-NEXT: Region tree:
; REGIONS-NEXT: [0] entry => <Function Return>
; REGIONS-NEXT: {{ *}}[1] header => A
; REGIONS-NEXT: End region tree
; REGIONS-LABEL: Region Tree for function: toplevel
; REGIONS-NEXT: Region tree:
; REGIONS-NEXT: [0] entry => <Function Return>
; REGIONS-NEXT: {{ *}}[1] header => A
; REGIONS-NEXT: {{ *}}[2] header => Flow
; REGIONS-NEXT: End region tree

define amdgpu_kernel void @toplevel(ptr addrspace(1) %p, ptr addrspace(1) %out, i32 %x) {
; CHECK-LABEL: @toplevel(
; CHECK:       header:
; CHECK:         %d = icmp ult i32 %u, %tid
; CHECK-NEXT:    br i1 %d, label %B, label %[[FLOW:[a-zA-Z0-9]+]]
; CHECK:       B:
; CHECK-NEXT:    store volatile i32 0, ptr addrspace(1) %p
; CHECK-NEXT:    br label %[[FLOW]]
; CHECK:       [[FLOW]]:
; CHECK-NEXT:    %[[EXIT:[0-9]+]] = phi i1 [ false, %B ], [ true, %header ]
; CHECK-NEXT:    br i1 %[[EXIT]], label %A, label %header
; CHECK:       A:
; CHECK-NEXT:    switch i32 %x, label %header [
;
; R600-LABEL: define amdgpu_kernel void @toplevel(
; R600:       header:
; R600:         %d = icmp ult i32 %u, %tid
; R600-NEXT:    br i1 %d, label %B, label %[[R600_FLOW:[a-zA-Z0-9]+]]
; R600:       B:
; R600-NEXT:    store volatile i32 0, ptr addrspace(1) %p
; R600-NEXT:    br label %[[R600_FLOW]]
; R600:       [[R600_FLOW]]:
; R600-NEXT:    %[[R600_EXIT:[0-9]+]] = phi i1 [ false, %B ], [ true, %header ]
; R600-NEXT:    br i1 %[[R600_EXIT]], label %A, label %header
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %u = load volatile i32, ptr addrspace(1) %p
  %d = icmp ult i32 %u, %tid
  br i1 %d, label %B, label %A

B:
  store volatile i32 0, ptr addrspace(1) %p
  br label %header

A:
  switch i32 %x, label %header [
    i32 0, label %R1
    i32 1, label %R2
  ]

R1:
  %gep = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %u, ptr addrspace(1) %gep
  ret void

R2:
  ret void
}
