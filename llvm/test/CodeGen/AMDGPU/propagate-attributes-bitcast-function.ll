; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN: foo1:
; v_cndmask_b32_e64 v0, 0, 1, vcc_lo{{$}}
; GCN: kernel1:
; GCN: foo1@gotpcrel32@lo+4
; GCN: foo1@gotpcrel32@hi+12

define void @foo1(i32 %x) #1 {
entry:
  %cc = icmp eq i32 %x, 0
  store volatile i1 %cc, i1* undef
  ret void
}

define amdgpu_kernel void @kernel1(float %x) #0 {
entry:
  call void bitcast (void (i32)* @foo1 to void (float)*)(float %x)
  ret void
}

attributes #0 = { nounwind "target-features"="+wavefrontsize32" }
attributes #1 = { noinline nounwind "target-features"="+wavefrontsize64" }
