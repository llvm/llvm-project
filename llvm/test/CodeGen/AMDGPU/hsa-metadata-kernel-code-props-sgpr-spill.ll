; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa -enable-misched=0 -amdgpu-stress-sgpr=20 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX700 %s
; RUN: llc --amdgpu-xnack=false -mtriple=amdgpu8.03-amd-amdhsa -enable-misched=0 -amdgpu-stress-sgpr=20 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX803 %s
; RUN: llc --amdgpu-xnack=false -mtriple=amdgpu9.00-amd-amdhsa -enable-misched=0 -amdgpu-stress-sgpr=20 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX900 %s
; RUN: llc --amdgpu-xnack=false -mtriple=amdgpu10.10-amd-amdhsa -enable-misched=0 -amdgpu-stress-sgpr=20 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX1010 %s

; CHECK:   .name:       num_spilled_sgprs
; GFX700:   .sgpr_spill_count: 10
; GFX803:   .sgpr_spill_count: 10
; GFX900:   .sgpr_spill_count: 62
; GFX1010:  .sgpr_spill_count: 60
; CHECK:   .symbol:     num_spilled_sgprs.kd
define amdgpu_kernel void @num_spilled_sgprs(
    ptr addrspace(1) %out0, ptr addrspace(1) %out1, [8 x i32],
    ptr addrspace(1) %out2, ptr addrspace(1) %out3, [8 x i32],
    ptr addrspace(1) %out4, ptr addrspace(1) %out5, [8 x i32],
    ptr addrspace(1) %out6, ptr addrspace(1) %out7, [8 x i32],
    ptr addrspace(1) %out8, ptr addrspace(1) %out9, [8 x i32],
    ptr addrspace(1) %outa, ptr addrspace(1) %outb, [8 x i32],
    ptr addrspace(1) %outc, ptr addrspace(1) %outd, [8 x i32],
    ptr addrspace(1) %oute, ptr addrspace(1) %outf, [8 x i32],
    ptr addrspace(1) %outg, ptr addrspace(1) %outh, [8 x i32],
    ptr addrspace(1) %outi, ptr addrspace(1) %outj, [8 x i32],
    ptr addrspace(1) %outk, ptr addrspace(1) %outl, [8 x i32],
    ptr addrspace(1) %outm, ptr addrspace(1) %outn, [8 x i32],
    i32 %in0, i32 %in1, i32 %in2, i32 %in3, [8 x i32],
    i32 %in4, i32 %in5, i32 %in6, i32 %in7, [8 x i32],
    i32 %in8, i32 %in9, i32 %ina, i32 %inb, [8 x i32],
    i32 %inc, i32 %ind, i32 %ine, i32 %inf, i32 %ing, i32 %inh,
    i32 %ini, i32 %inj, i32 %ink) "amdgpu-no-flat-scratch-init" {
entry:
  store volatile i32 %in0, ptr addrspace(1) %out0
  store volatile i32 %in1, ptr addrspace(1) %out1
  store volatile i32 %in2, ptr addrspace(1) %out2
  store volatile i32 %in3, ptr addrspace(1) %out3
  store volatile i32 %in4, ptr addrspace(1) %out4
  store volatile i32 %in5, ptr addrspace(1) %out5
  store volatile i32 %in6, ptr addrspace(1) %out6
  store volatile i32 %in7, ptr addrspace(1) %out7
  store volatile i32 %in8, ptr addrspace(1) %out8
  store volatile i32 %in9, ptr addrspace(1) %out9
  store volatile i32 %ina, ptr addrspace(1) %outa
  store volatile i32 %inb, ptr addrspace(1) %outb
  store volatile i32 %inc, ptr addrspace(1) %outc
  store volatile i32 %ind, ptr addrspace(1) %outd
  store volatile i32 %ine, ptr addrspace(1) %oute
  store volatile i32 %inf, ptr addrspace(1) %outf
  store volatile i32 %ing, ptr addrspace(1) %outg
  store volatile i32 %inh, ptr addrspace(1) %outh
  store volatile i32 %ini, ptr addrspace(1) %outi
  store volatile i32 %inj, ptr addrspace(1) %outj
  store volatile i32 %ink, ptr addrspace(1) %outk
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
