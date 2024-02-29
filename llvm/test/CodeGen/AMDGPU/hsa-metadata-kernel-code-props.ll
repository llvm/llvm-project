; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX700,WAVE64 %s
; RUN: llc -mattr=-xnack -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX803,WAVE64 %s
; RUN: llc -mattr=-xnack -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX900,WAVE64 %s
; RUN: llc -mattr=-xnack -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX1010,WAVE32 %s

@var = addrspace(1) global float 0.0

; CHECK: ---
; CHECK:  amdhsa.kernels:

; CHECK:   - .args:
; CHECK:     .group_segment_fixed_size: 0
; CHECK:     .kernarg_segment_align: 8
; CHECK:     .kernarg_segment_size: 24
; CHECK:     .max_flat_workgroup_size: 1024
; CHECK:     .name:           test
; CHECK:     .private_segment_fixed_size: 0
; CHECK:     .sgpr_count:     6
; CHECK:     .symbol:         test.kd
; CHECK:     .vgpr_count:     {{3|6}}
; WAVE64:    .wavefront_size: 64
; WAVE32:    .wavefront_size: 32
define amdgpu_kernel void @test(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:   - .args:
; CHECK:     .max_flat_workgroup_size: 256
define amdgpu_kernel void @test_max_flat_workgroup_size(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) #2 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:   .name:       num_spilled_sgprs
; GFX700:   .sgpr_spill_count: 12
; GFX803:   .sgpr_spill_count: 12
; GFX900:   .sgpr_spill_count: 48
; GFX1010:  .sgpr_spill_count: 48
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
    i32 %in0, i32 %in1, i32 %in2, i32 %in3, [8 x i32],
    i32 %in4, i32 %in5, i32 %in6, i32 %in7, [8 x i32],
    i32 %in8, i32 %in9, i32 %ina, i32 %inb, [8 x i32],
    i32 %inc, i32 %ind, i32 %ine, i32 %inf) #0 {
entry:
  store i32 %in0, ptr addrspace(1) %out0
  store i32 %in1, ptr addrspace(1) %out1
  store i32 %in2, ptr addrspace(1) %out2
  store i32 %in3, ptr addrspace(1) %out3
  store i32 %in4, ptr addrspace(1) %out4
  store i32 %in5, ptr addrspace(1) %out5
  store i32 %in6, ptr addrspace(1) %out6
  store i32 %in7, ptr addrspace(1) %out7
  store i32 %in8, ptr addrspace(1) %out8
  store i32 %in9, ptr addrspace(1) %out9
  store i32 %ina, ptr addrspace(1) %outa
  store i32 %inb, ptr addrspace(1) %outb
  store i32 %inc, ptr addrspace(1) %outc
  store i32 %ind, ptr addrspace(1) %outd
  store i32 %ine, ptr addrspace(1) %oute
  store i32 %inf, ptr addrspace(1) %outf
  ret void
}

; CHECK:   .name:       num_spilled_vgprs
; CHECK:   .symbol:     num_spilled_vgprs.kd
; CHECK:   .vgpr_spill_count: {{13|14}}
define amdgpu_kernel void @num_spilled_vgprs() #1 {
  %val0 = load volatile float, ptr addrspace(1) @var
  %val1 = load volatile float, ptr addrspace(1) @var
  %val2 = load volatile float, ptr addrspace(1) @var
  %val3 = load volatile float, ptr addrspace(1) @var
  %val4 = load volatile float, ptr addrspace(1) @var
  %val5 = load volatile float, ptr addrspace(1) @var
  %val6 = load volatile float, ptr addrspace(1) @var
  %val7 = load volatile float, ptr addrspace(1) @var
  %val8 = load volatile float, ptr addrspace(1) @var
  %val9 = load volatile float, ptr addrspace(1) @var
  %val10 = load volatile float, ptr addrspace(1) @var
  %val11 = load volatile float, ptr addrspace(1) @var
  %val12 = load volatile float, ptr addrspace(1) @var
  %val13 = load volatile float, ptr addrspace(1) @var
  %val14 = load volatile float, ptr addrspace(1) @var
  %val15 = load volatile float, ptr addrspace(1) @var
  %val16 = load volatile float, ptr addrspace(1) @var
  %val17 = load volatile float, ptr addrspace(1) @var
  %val18 = load volatile float, ptr addrspace(1) @var
  %val19 = load volatile float, ptr addrspace(1) @var
  %val20 = load volatile float, ptr addrspace(1) @var
  %val21 = load volatile float, ptr addrspace(1) @var
  %val22 = load volatile float, ptr addrspace(1) @var
  %val23 = load volatile float, ptr addrspace(1) @var
  %val24 = load volatile float, ptr addrspace(1) @var
  %val25 = load volatile float, ptr addrspace(1) @var
  %val26 = load volatile float, ptr addrspace(1) @var
  %val27 = load volatile float, ptr addrspace(1) @var
  %val28 = load volatile float, ptr addrspace(1) @var
  %val29 = load volatile float, ptr addrspace(1) @var
  %val30 = load volatile float, ptr addrspace(1) @var

  store volatile float %val0, ptr addrspace(1) @var
  store volatile float %val1, ptr addrspace(1) @var
  store volatile float %val2, ptr addrspace(1) @var
  store volatile float %val3, ptr addrspace(1) @var
  store volatile float %val4, ptr addrspace(1) @var
  store volatile float %val5, ptr addrspace(1) @var
  store volatile float %val6, ptr addrspace(1) @var
  store volatile float %val7, ptr addrspace(1) @var
  store volatile float %val8, ptr addrspace(1) @var
  store volatile float %val9, ptr addrspace(1) @var
  store volatile float %val10, ptr addrspace(1) @var
  store volatile float %val11, ptr addrspace(1) @var
  store volatile float %val12, ptr addrspace(1) @var
  store volatile float %val13, ptr addrspace(1) @var
  store volatile float %val14, ptr addrspace(1) @var
  store volatile float %val15, ptr addrspace(1) @var
  store volatile float %val16, ptr addrspace(1) @var
  store volatile float %val17, ptr addrspace(1) @var
  store volatile float %val18, ptr addrspace(1) @var
  store volatile float %val19, ptr addrspace(1) @var
  store volatile float %val20, ptr addrspace(1) @var
  store volatile float %val21, ptr addrspace(1) @var
  store volatile float %val22, ptr addrspace(1) @var
  store volatile float %val23, ptr addrspace(1) @var
  store volatile float %val24, ptr addrspace(1) @var
  store volatile float %val25, ptr addrspace(1) @var
  store volatile float %val26, ptr addrspace(1) @var
  store volatile float %val27, ptr addrspace(1) @var
  store volatile float %val28, ptr addrspace(1) @var
  store volatile float %val29, ptr addrspace(1) @var
  store volatile float %val30, ptr addrspace(1) @var

  ret void
}

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 1

attributes #0 = { "amdgpu-num-sgpr"="14" }
attributes #1 = { "amdgpu-num-vgpr"="20" }
attributes #2 = { "amdgpu-flat-work-group-size"="1,256" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
