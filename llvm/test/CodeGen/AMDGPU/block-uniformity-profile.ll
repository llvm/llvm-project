; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O0 -stop-after=finalize-isel -o - %s | \
; RUN:   llc -mtriple=amdgcn-amd-amdhsa -passes='print<block-uniformity-profile>' -x mir -filetype=null 2>&1 | FileCheck %s

; Test that BlockUniformityProfileProxy correctly reads block-uniformity-profile
; metadata from IR basic blocks and classifies machine blocks.
;
; This metadata is attached during PGO-use phase to indicate whether a basic block
; was executed uniformly (all lanes together) or divergently (partial wave).
;
; The analysis is consumed by SpillPlacement to flatten block frequencies for
; divergent blocks, preventing PGO from causing regressions on divergent code paths.

; CHECK-LABEL: BlockUniformityProfile for function: @uniform_blocks
; CHECK-NEXT: HasProfile: true
; CHECK: %bb.{{[0-9]+}} (%entry): uniform
define amdgpu_kernel void @uniform_blocks(ptr addrspace(1) %out) #0 {
entry:
  store i32 1, ptr addrspace(1) %out, align 4
  ret void, !block-uniformity-profile !0
}

; CHECK-LABEL: BlockUniformityProfile for function: @divergent_blocks
; CHECK-NEXT: HasProfile: true
; CHECK-DAG: %bb.{{[0-9]+}} (%if.then): divergent
; CHECK-DAG: %bb.{{[0-9]+}} (%if.else): uniform
define amdgpu_kernel void @divergent_blocks(ptr addrspace(1) %out, i32 %tid) #0 {
entry:
  %cmp = icmp eq i32 %tid, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  store i32 1, ptr addrspace(1) %out, align 4
  ret void, !block-uniformity-profile !1

if.else:
  store i32 2, ptr addrspace(1) %out, align 4
  ret void, !block-uniformity-profile !0
}

; CHECK-LABEL: BlockUniformityProfile for function: @missing_metadata
; CHECK-NEXT: HasProfile: true
; CHECK-DAG: %bb.{{[0-9]+}} (%if.then): no PGO annotation (treated divergent for spill placement)
; CHECK-DAG: %bb.{{[0-9]+}} (%if.else): uniform
define amdgpu_kernel void @missing_metadata(ptr addrspace(1) %out, i32 %cond) #0 {
entry:
  %cmp = icmp sgt i32 %cond, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  store i32 1, ptr addrspace(1) %out, align 4
  ret void

if.else:
  store i32 2, ptr addrspace(1) %out, align 4
  ret void, !block-uniformity-profile !0
}

; CHECK-LABEL: BlockUniformityProfile for function: @no_divergence_metadata
; CHECK-NEXT: HasProfile: false
define amdgpu_kernel void @no_divergence_metadata(ptr addrspace(1) %out, i32 %cond) #0 {
entry:
  ; No uniformity metadata - analysis should report hasProfile() = false
  %cmp = icmp sgt i32 %cond, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  store i32 1, ptr addrspace(1) %out, align 4
  ret void

if.else:
  store i32 2, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }

; Metadata: i1 true = uniform, i1 false = divergent
!0 = !{i1 true}   ; uniform
!1 = !{i1 false}  ; divergent
