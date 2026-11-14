; RUN: llc -global-isel -mtriple=amdgpu8.02 -stop-after=legalizer -verify-machineinstrs -o - %s | FileCheck %s
; RUN: llc -global-isel -mtriple=amdgpu9.00 -stop-after=legalizer -verify-machineinstrs -o - %s | FileCheck %s

; With TFE enabled, both packed and unpacked D16 loads return an i32 data
; component followed by the status. Recover the half value through i16.
define amdgpu_ps half @image_load_tfe_f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t, ptr addrspace(1) %out) {
; CHECK-LABEL: name: image_load_tfe_f16
; CHECK: [[LOAD:%[0-9]+]]:_(<2 x i32>) = G_AMDGPU_INTRIN_IMAGE_LOAD_D16
; CHECK-NEXT: [[DATA:%[0-9]+]]:_(i32), [[STATUS:%[0-9]+]]:_(i32) = G_UNMERGE_VALUES [[LOAD]](<2 x i32>)
; CHECK-NEXT: [[BITS:%[0-9]+]]:_(i16) = G_TRUNC [[DATA]](i32)
; CHECK-NEXT: {{%[0-9]+}}:_(f16) = G_BITCAST [[BITS]](i16)
; CHECK-NEXT: G_STORE [[STATUS]](i32), {{%[0-9]+}}(p1)
  %res = call { half, i32 } @llvm.amdgcn.image.load.2d.sl_f16i32s.i32(i32 1, i32 %s, i32 %t, <8 x i32> %rsrc, i32 1, i32 0)
  %tex = extractvalue { half, i32 } %res, 0
  %tfe = extractvalue { half, i32 } %res, 1
  store i32 %tfe, ptr addrspace(1) %out
  ret half %tex
}
