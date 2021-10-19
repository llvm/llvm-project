; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=tahiti | FileCheck --check-prefixes=PAL,CI --enable-var-scope %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=tonga | FileCheck --check-prefixes=PAL,VI --enable-var-scope %s

; PAL-NOT: .AMDGPU.config
; PAL-LABEL: {{^}}simple:
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

define amdgpu_kernel void @scratch(<2 x i32> %in, i32 %idx, i32 addrspace(5)* %out) {
entry:
  %v = alloca [2 x i32], addrspace(5)
  %vv = bitcast [2 x i32] addrspace(5)* %v to <2 x i32> addrspace(5)*
  store <2 x i32> %in, <2 x i32> addrspace(5)* %vv
  %e = getelementptr [2 x i32], [2 x i32] addrspace(5)* %v, i32 0, i32 %idx
  %x = load i32, i32 addrspace(5)* %e
  store i32 %x, i32 addrspace(5)* %out
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

define amdgpu_kernel void @scratch2(<2 x i32> %in, i32 %idx, i32 addrspace(5)* %out) #0 {
entry:
  %v = alloca [2 x i32], addrspace(5)
  %vv = bitcast [2 x i32] addrspace(5)* %v to <2 x i32> addrspace(5)*
  store <2 x i32> %in, <2 x i32> addrspace(5)* %vv
  %e = getelementptr [2 x i32], [2 x i32] addrspace(5)* %v, i32 0, i32 %idx
  %x = load i32, i32 addrspace(5)* %e
  store i32 %x, i32 addrspace(5)* %out
  ret void
}

; Check code sequence for amdpal use of scratch for alloca in a compute shader.
; The scratch descriptor is loaded from offset 0x10 of the GIT, rather than offset
; 0 in a graphics shader.
; Prior to GCN3 s_load_dword offsets are dwords, so the offset will be 0x4.

; PAL-LABEL: {{^}}scratch2_cs:
; PAL: s_movk_i32 s{{[0-9]+}}, 0x1234
; PAL: s_mov_b32 s[[GITPTR:[0-9]+]], s0
; CI: s_load_dwordx4 s{{\[}}[[SCRATCHDESC:[0-9]+]]:{{[0-9]+]}}, s{{\[}}[[GITPTR]]:{{[0-9]+\]}}, 0x4
; VI: s_load_dwordx4 s{{\[}}[[SCRATCHDESC:[0-9]+]]:{{[0-9]+]}}, s{{\[}}[[GITPTR]]:{{[0-9]+\]}}, 0x10
; PAL: buffer_store{{.*}}, s{{\[}}[[SCRATCHDESC]]:

define amdgpu_cs void @scratch2_cs(i32 inreg, i32 inreg, i32 inreg, <3 x i32> inreg, i32 inreg, <3 x i32> %coord, <2 x i32> %in, i32 %extra, i32 %idx) #0 {
entry:
  %v = alloca [3 x i32], addrspace(5)
  %v0 = getelementptr [3 x i32], [3 x i32] addrspace(5)* %v, i32 0, i32 0
  %v1 = getelementptr [3 x i32], [3 x i32] addrspace(5)* %v, i32 0, i32 1
  store i32 %extra, i32 addrspace(5)* %v0
  %v1a = bitcast i32 addrspace(5)* %v1 to [2 x i32] addrspace(5)*
  %vv = bitcast [2 x i32] addrspace(5)* %v1a to <2 x i32> addrspace(5)*
  store <2 x i32> %in, <2 x i32> addrspace(5)* %vv
  %e = getelementptr [2 x i32], [2 x i32] addrspace(5)* %v1a, i32 0, i32 %idx
  %x = load i32, i32 addrspace(5)* %e
  %xf = bitcast i32 %x to float
  call void @llvm.amdgcn.raw.buffer.store.f32(float %xf, <4 x i32> undef, i32 0, i32 0, i32 0)
  ret void
}

attributes #0 = { nounwind "amdgpu-git-ptr-high"="0x1234" }

declare void @llvm.amdgcn.raw.buffer.store.f32(float, <4 x i32>, i32, i32, i32 immarg)


; PAL:         .amdgpu_pal_metadata
; PAL-NEXT: ---
; PAL-NEXT: amdpal.pipelines:
; PAL-NEXT:   - .hardware_stages:
; PAL-NEXT:       .cs:
; PAL-NEXT:         .entry_point:    scratch2_cs
; PAL-NEXT:         .scratch_memory_size: 0x10
; PAL-NEXT:         .sgpr_count:     0x
; PAL-NEXT:         .vgpr_count:     0x
