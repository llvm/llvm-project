; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=-code-object-v3 -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -mattr=-code-object-v3 -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes | FileCheck --check-prefix=CHECK --check-prefix=GFX803 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

@var = addrspace(1) global float 0.0

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]
; CHECK:  Kernels:

; CHECK-LABEL: - Name:       test
; CHECK:   SymbolName: 'test@kd'
; CHECK:   CodeProps:
; CHECK:     KernargSegmentSize:      24
; CHECK:     GroupSegmentFixedSize:   0
; CHECK:     PrivateSegmentFixedSize: 0
; CHECK:     KernargSegmentAlign:     8
; CHECK:     WavefrontSize:           64
; CHECK:     NumSGPRs:                8
; CHECK:     NumVGPRs:                6
; CHECK:     MaxFlatWorkGroupSize:    256
define amdgpu_kernel void @test(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK-LABEL: - Name:       num_spilled_sgprs
; CHECK:   SymbolName: 'num_spilled_sgprs@kd'
; CHECK:   CodeProps:
; GFX700:     NumSpilledSGPRs: 40
; GFX803:     NumSpilledSGPRs: 24
; GFX900:     NumSpilledSGPRs: 24
define amdgpu_kernel void @num_spilled_sgprs(
    i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, [8 x i32],
    i32 addrspace(1)* %out2, i32 addrspace(1)* %out3, [8 x i32],
    i32 addrspace(1)* %out4, i32 addrspace(1)* %out5, [8 x i32],
    i32 addrspace(1)* %out6, i32 addrspace(1)* %out7, [8 x i32],
    i32 addrspace(1)* %out8, i32 addrspace(1)* %out9, [8 x i32],
    i32 addrspace(1)* %outa, i32 addrspace(1)* %outb, [8 x i32],
    i32 addrspace(1)* %outc, i32 addrspace(1)* %outd, [8 x i32],
    i32 addrspace(1)* %oute, i32 addrspace(1)* %outf, [8 x i32],
    i32 %in0, i32 %in1, i32 %in2, i32 %in3, [8 x i32],
    i32 %in4, i32 %in5, i32 %in6, i32 %in7, [8 x i32],
    i32 %in8, i32 %in9, i32 %ina, i32 %inb, [8 x i32],
    i32 %inc, i32 %ind, i32 %ine, i32 %inf) #0 {
entry:
  store i32 %in0, i32 addrspace(1)* %out0
  store i32 %in1, i32 addrspace(1)* %out1
  store i32 %in2, i32 addrspace(1)* %out2
  store i32 %in3, i32 addrspace(1)* %out3
  store i32 %in4, i32 addrspace(1)* %out4
  store i32 %in5, i32 addrspace(1)* %out5
  store i32 %in6, i32 addrspace(1)* %out6
  store i32 %in7, i32 addrspace(1)* %out7
  store i32 %in8, i32 addrspace(1)* %out8
  store i32 %in9, i32 addrspace(1)* %out9
  store i32 %ina, i32 addrspace(1)* %outa
  store i32 %inb, i32 addrspace(1)* %outb
  store i32 %inc, i32 addrspace(1)* %outc
  store i32 %ind, i32 addrspace(1)* %outd
  store i32 %ine, i32 addrspace(1)* %oute
  store i32 %inf, i32 addrspace(1)* %outf
  ret void
}

; CHECK-LABEL: - Name:       num_spilled_vgprs
; CHECK:   SymbolName: 'num_spilled_vgprs@kd'
; CHECK:   CodeProps:
; CHECK:     NumSpilledVGPRs: 14
define amdgpu_kernel void @num_spilled_vgprs() #1 {
  %val0 = load volatile float, float addrspace(1)* @var
  %val1 = load volatile float, float addrspace(1)* @var
  %val2 = load volatile float, float addrspace(1)* @var
  %val3 = load volatile float, float addrspace(1)* @var
  %val4 = load volatile float, float addrspace(1)* @var
  %val5 = load volatile float, float addrspace(1)* @var
  %val6 = load volatile float, float addrspace(1)* @var
  %val7 = load volatile float, float addrspace(1)* @var
  %val8 = load volatile float, float addrspace(1)* @var
  %val9 = load volatile float, float addrspace(1)* @var
  %val10 = load volatile float, float addrspace(1)* @var
  %val11 = load volatile float, float addrspace(1)* @var
  %val12 = load volatile float, float addrspace(1)* @var
  %val13 = load volatile float, float addrspace(1)* @var
  %val14 = load volatile float, float addrspace(1)* @var
  %val15 = load volatile float, float addrspace(1)* @var
  %val16 = load volatile float, float addrspace(1)* @var
  %val17 = load volatile float, float addrspace(1)* @var
  %val18 = load volatile float, float addrspace(1)* @var
  %val19 = load volatile float, float addrspace(1)* @var
  %val20 = load volatile float, float addrspace(1)* @var
  %val21 = load volatile float, float addrspace(1)* @var
  %val22 = load volatile float, float addrspace(1)* @var
  %val23 = load volatile float, float addrspace(1)* @var
  %val24 = load volatile float, float addrspace(1)* @var
  %val25 = load volatile float, float addrspace(1)* @var
  %val26 = load volatile float, float addrspace(1)* @var
  %val27 = load volatile float, float addrspace(1)* @var
  %val28 = load volatile float, float addrspace(1)* @var
  %val29 = load volatile float, float addrspace(1)* @var
  %val30 = load volatile float, float addrspace(1)* @var

  store volatile float %val0, float addrspace(1)* @var
  store volatile float %val1, float addrspace(1)* @var
  store volatile float %val2, float addrspace(1)* @var
  store volatile float %val3, float addrspace(1)* @var
  store volatile float %val4, float addrspace(1)* @var
  store volatile float %val5, float addrspace(1)* @var
  store volatile float %val6, float addrspace(1)* @var
  store volatile float %val7, float addrspace(1)* @var
  store volatile float %val8, float addrspace(1)* @var
  store volatile float %val9, float addrspace(1)* @var
  store volatile float %val10, float addrspace(1)* @var
  store volatile float %val11, float addrspace(1)* @var
  store volatile float %val12, float addrspace(1)* @var
  store volatile float %val13, float addrspace(1)* @var
  store volatile float %val14, float addrspace(1)* @var
  store volatile float %val15, float addrspace(1)* @var
  store volatile float %val16, float addrspace(1)* @var
  store volatile float %val17, float addrspace(1)* @var
  store volatile float %val18, float addrspace(1)* @var
  store volatile float %val19, float addrspace(1)* @var
  store volatile float %val20, float addrspace(1)* @var
  store volatile float %val21, float addrspace(1)* @var
  store volatile float %val22, float addrspace(1)* @var
  store volatile float %val23, float addrspace(1)* @var
  store volatile float %val24, float addrspace(1)* @var
  store volatile float %val25, float addrspace(1)* @var
  store volatile float %val26, float addrspace(1)* @var
  store volatile float %val27, float addrspace(1)* @var
  store volatile float %val28, float addrspace(1)* @var
  store volatile float %val29, float addrspace(1)* @var
  store volatile float %val30, float addrspace(1)* @var

  ret void
}

attributes #0 = { "amdgpu-num-sgpr"="14" }
attributes #1 = { "amdgpu-num-vgpr"="20" }
