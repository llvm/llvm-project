; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -amdgpu-enable-lower-module-lds=0 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN %s
; FIXME: Merge with DAG test

@lds.external = external unnamed_addr addrspace(3) global [0 x i32]
@lds.defined = unnamed_addr addrspace(3) global [8 x i32] undef, align 8

; GCN-LABEL: {{^}}test_basic:
; GCN: s_add_u32 s0, lds.defined@abs32@lo, s0 ; encoding: [0xff,0x00,0x00,0x80,A,A,A,A]
; GCN: v_mov_b32_e32 v2, s0 ; encoding: [0x00,0x02,0x04,0x7e]

; GCN: .globl lds.external
; GCN: .amdgpu_lds lds.external, 0, 4
; GCN: .globl lds.defined
; GCN: .amdgpu_lds lds.defined, 32, 8
define amdgpu_gs float @test_basic(i32 inreg %wave, i32 %arg1) #0 {
main_body:
  %gep0 = getelementptr [0 x i32], [0 x i32] addrspace(3)* @lds.external, i32 0, i32 %arg1
  %tmp = load i32, i32 addrspace(3)* %gep0

  %gep1 = getelementptr [8 x i32], [8 x i32] addrspace(3)* @lds.defined, i32 0, i32 %wave
  store i32 123, i32 addrspace(3)* %gep1

  %r = bitcast i32 %tmp to float
  ret float %r
}

attributes #0 = { "no-signed-zeros-fp-math"="true" }
attributes #4 = { convergent nounwind readnone }
