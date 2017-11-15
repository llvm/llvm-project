; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=tahiti | FileCheck --check-prefix=PAL --enable-var-scope %s

; PAL: .AMDGPU.config

define amdgpu_kernel void @simple(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; Check code sequence for amdpal use of scratch for alloca. This is the case
; where the high half of the address comes from s_getpc.

; PAL-LABEL: {{^}}scratch:
; PAL: s_getpc_b64 s{{\[}}[[GITPTR:[0-9]+]]:
; PAL: s_mov_b32 s[[GITPTR]], s0
; PAL: s_load_dwordx4 s{{\[}}[[SCRATCHDESC:[0-9]+]]:{{[0-9]+]}}, s{{\[}}[[GITPTR]]:
; PAL: buffer_store{{.*}}, s{{\[}}[[SCRATCHDESC]]:

define amdgpu_kernel void @scratch(<2 x i32> %in, i32 %idx, i32* %out) {
entry:
  %v = alloca [2 x i32]
  %vv = bitcast [2 x i32]* %v to <2 x i32>*
  store <2 x i32> %in, <2 x i32>* %vv
  %e = getelementptr [2 x i32], [2 x i32]* %v, i32 0, i32 %idx
  %x = load i32, i32* %e
  store i32 %x, i32* %out
  ret void
}

; Check code sequence for amdpal use of scratch for alloca. This is the case
; where the amdgpu-git-ptr-high function attribute gives the high half of the
; address to use.
; Looks like you can't do arithmetic on a filecheck variable, so we can't test
; that the s_movk_i32 is into a reg that is one more than the following
; s_mov_b32.

; PAL-LABEL: {{^}}scratch2:
; PAL: s_movk_i32 s{{[0-9]+}}, 0x1234
; PAL: s_mov_b32 s[[GITPTR:[0-9]+]], s0
; PAL: s_load_dwordx4 s{{\[}}[[SCRATCHDESC:[0-9]+]]:{{[0-9]+]}}, s{{\[}}[[GITPTR]]:
; PAL: buffer_store{{.*}}, s{{\[}}[[SCRATCHDESC]]:

define amdgpu_kernel void @scratch2(<2 x i32> %in, i32 %idx, i32* %out) #0 {
entry:
  %v = alloca [2 x i32]
  %vv = bitcast [2 x i32]* %v to <2 x i32>*
  store <2 x i32> %in, <2 x i32>* %vv
  %e = getelementptr [2 x i32], [2 x i32]* %v, i32 0, i32 %idx
  %x = load i32, i32* %e
  store i32 %x, i32* %out
  ret void
}

attributes #0 = { nounwind "amdgpu-git-ptr-high"="0x1234" }

; Check we have CS_NUM_USED_VGPRS in PAL metadata.
; PAL: .amd_amdgpu_pal_metadata {{.*}},0x10000027,
