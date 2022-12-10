; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -O0 -amdgpu-ir-lower-kernel-arguments=0 -stop-after=irtranslator -global-isel %s -o - | FileCheck -check-prefix=HSA %s

; HSA-LABEL: name: default_kernel
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$vgpr0', virtual-reg: '%1' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%2' }
; HSA-NEXT: - { reg: '$sgpr5', virtual-reg: '%3' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @default_kernel() {
  ret void
}


; HSA-LABEL: name: workgroup_id_x{{$}}
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$vgpr0', virtual-reg: '%1' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%2' }
; HSA-NEXT: - { reg: '$sgpr5', virtual-reg: '%3' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @workgroup_id_x() {
  %id = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id, ptr addrspace(1) undef
  ret void
}

; HSA-LABEL: name: workgroup_id_y{{$}}
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$vgpr0', virtual-reg: '%1' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%2' }
; HSA-NEXT: - { reg: '$sgpr5', virtual-reg: '%3' }
; HSA-NEXT: - { reg: '$sgpr6', virtual-reg: '%4' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @workgroup_id_y() {
  %id = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %id, ptr addrspace(1) undef
  ret void
}

; HSA-LABEL: name: workgroup_id_z{{$}}
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$vgpr0', virtual-reg: '%1' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%2' }
; HSA-NEXT: - { reg: '$sgpr5', virtual-reg: '%3' }
; HSA-NEXT: - { reg: '$sgpr6', virtual-reg: '%4' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @workgroup_id_z() {
  %id = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %id, ptr addrspace(1) undef
  ret void
}

; HSA-LABEL: name: workgroup_id_xy{{$}}
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$vgpr0', virtual-reg: '%1' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%2' }
; HSA-NEXT: - { reg: '$sgpr5', virtual-reg: '%3' }
; HSA-NEXT: - { reg: '$sgpr6', virtual-reg: '%4' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @workgroup_id_xy() {
  %id0 = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id0, ptr addrspace(1) undef
  %id1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %id1, ptr addrspace(1) undef
  ret void
}

; HSA-LABEL: name: workgroup_id_xyz{{$}}
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$vgpr0', virtual-reg: '%1' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%2' }
; HSA-NEXT: - { reg: '$sgpr5', virtual-reg: '%3' }
; HSA-NEXT: - { reg: '$sgpr6', virtual-reg: '%4' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @workgroup_id_xyz() {
  %id0 = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id0, ptr addrspace(1) undef
  %id1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %id1, ptr addrspace(1) undef
  %id2 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %id2, ptr addrspace(1) undef
  ret void
}

; HSA-LABEL: name: workgroup_id_yz{{$}}
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$vgpr0', virtual-reg: '%1' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%2' }
; HSA-NEXT: - { reg: '$sgpr5', virtual-reg: '%3' }
; HSA-NEXT: - { reg: '$sgpr6', virtual-reg: '%4' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @workgroup_id_yz() {
  %id0 = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id0, ptr addrspace(1) undef
  %id1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %id1, ptr addrspace(1) undef
  ret void
}

; HSA-LABEL: name: workgroup_id_xz{{$}}
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$vgpr0', virtual-reg: '%1' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%2' }
; HSA-NEXT: - { reg: '$sgpr5', virtual-reg: '%3' }
; HSA-NEXT: - { reg: '$sgpr6', virtual-reg: '%4' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @workgroup_id_xz() {
  %id0 = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id0, ptr addrspace(1) undef
  %id1 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %id1, ptr addrspace(1) undef
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

attributes #0 = { nounwind readnone speculatable }
