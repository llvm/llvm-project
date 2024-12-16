; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -vgpr-regalloc=greedy -filetype=null %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -vgpr-regalloc=basic -filetype=null %s 2>&1 | FileCheck %s

; TODO: Check regalloc fast when it doesn't assert after failing.

; CHECK: LLVM ERROR: no registers from class available to allocate

declare <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32, i32, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)

define <32 x i32> @no_registers_from_class_available_to_allocate(<32 x i32> %arg) #0 {
  %ret = call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 1, i32 2, <32 x i32> %arg, i32 1, i32 2, i32 3)
  ret <32 x i32> %ret
}

; FIXME: Special case in fast RA, asserts. Also asserts in greedy
; define void @no_registers_from_class_available_to_allocate_undef_asm() #0 {
;   call void asm sideeffect "; use $0", "v"(<32 x i32> poison)
;   ret void
; }

attributes #0 = { "amdgpu-waves-per-eu"="10,10" }
