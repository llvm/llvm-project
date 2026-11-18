; RUN: llc -global-isel -mtriple=amdgpu8.02 -stop-after=legalizer -verify-machineinstrs -o - %s | FileCheck %s --check-prefixes=GCN,UNPACKED
; RUN: llc -global-isel -mtriple=amdgpu9.00 -stop-after=legalizer -verify-machineinstrs -o - %s | FileCheck %s --check-prefixes=GCN,PACKED

define amdgpu_ps half @buffer_load_half(<4 x i32> inreg %rsrc) {
; GCN-LABEL: name: buffer_load_half
; GCN: [[LOAD:%[0-9]+]]:_(i32) = G_AMDGPU_BUFFER_LOAD_USHORT
; GCN-NEXT: [[TRUNC:%[0-9]+]]:_(i16) = G_TRUNC [[LOAD]](i32)
; GCN-NEXT: {{%[0-9]+}}:_(f16) = G_BITCAST [[TRUNC]](i16)
  %x = call half @llvm.amdgcn.raw.buffer.load.f16(<4 x i32> %rsrc, i32 0, i32 0, i32 0)
  ret half %x
}

define amdgpu_ps half @buffer_load_format_half(<4 x i32> inreg %rsrc) {
; GCN-LABEL: name: buffer_load_format_half
; GCN: [[LOAD:%[0-9]+]]:_(i32) = G_AMDGPU_BUFFER_LOAD_FORMAT_D16
; GCN-NEXT: [[TRUNC:%[0-9]+]]:_(i16) = G_TRUNC [[LOAD]](i32)
; GCN-NEXT: {{%[0-9]+}}:_(f16) = G_BITCAST [[TRUNC]](i16)
  %x = call half @llvm.amdgcn.raw.buffer.load.format.f16(<4 x i32> %rsrc, i32 0, i32 0, i32 0)
  ret half %x
}

define amdgpu_ps <2 x half> @buffer_load_format_v2half(<4 x i32> inreg %rsrc) {
; GCN-LABEL: name: buffer_load_format_v2half
; UNPACKED: [[LOAD:%[0-9]+]]:_(<2 x i32>) = G_AMDGPU_BUFFER_LOAD_FORMAT_D16
; UNPACKED-NEXT: [[LO:%[0-9]+]]:_(i32), [[HI:%[0-9]+]]:_(i32) = G_UNMERGE_VALUES [[LOAD]](<2 x i32>)
; UNPACKED-NEXT: [[MASK:%[0-9]+]]:_(i32) = G_CONSTANT i32 65535
; UNPACKED-NEXT: [[LO16:%[0-9]+]]:_(i32) = G_AND [[LO]], [[MASK]]
; UNPACKED-NEXT: [[HI16:%[0-9]+]]:_(i32) = G_AND [[HI]], [[MASK]]
; UNPACKED-NEXT: [[SHIFT:%[0-9]+]]:_(i32) = G_CONSTANT i32 16
; UNPACKED-NEXT: [[HIGH:%[0-9]+]]:_(i32) = G_SHL [[HI16]], [[SHIFT]](i32)
; UNPACKED-NEXT: [[PACK:%[0-9]+]]:_(i32) = G_OR [[LO16]], [[HIGH]]
; UNPACKED-NEXT: [[BITS:%[0-9]+]]:_(<2 x i16>) = G_BITCAST [[PACK]](i32)
; UNPACKED-NEXT: {{%[0-9]+}}:_(<2 x f16>) = G_BITCAST [[BITS]](<2 x i16>)
; PACKED: {{%[0-9]+}}:_(<2 x f16>) = G_AMDGPU_BUFFER_LOAD_FORMAT_D16
  %x = call <2 x half> @llvm.amdgcn.raw.buffer.load.format.v2f16(<4 x i32> %rsrc, i32 0, i32 0, i32 0)
  ret <2 x half> %x
}

declare half @llvm.amdgcn.raw.buffer.load.f16(<4 x i32>, i32, i32, i32 immarg)
declare half @llvm.amdgcn.raw.buffer.load.format.f16(<4 x i32>, i32, i32, i32 immarg)
declare <2 x half> @llvm.amdgcn.raw.buffer.load.format.v2f16(<4 x i32>, i32, i32, i32 immarg)
