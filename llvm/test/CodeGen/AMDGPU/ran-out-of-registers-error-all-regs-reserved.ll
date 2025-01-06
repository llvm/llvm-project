; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -vgpr-regalloc=greedy -verify-machineinstrs=0 -filetype=null %s 2>&1 | FileCheck -implicit-check-not=error %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -vgpr-regalloc=basic -verify-machineinstrs=0 -filetype=null %s 2>&1 | FileCheck -implicit-check-not=error %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -vgpr-regalloc=fast -verify-machineinstrs=0 -filetype=null %s 2>&1 | FileCheck -implicit-check-not=error %s

; FIXME: Should pass verifier after failure.

declare <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32, i32, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)

; CHECK: error: <unknown>:0:0: no registers from class available to allocate in function 'no_registers_from_class_available_to_allocate'
define <32 x i32> @no_registers_from_class_available_to_allocate(<32 x i32> %arg) #0 {
  %ret = call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 1, i32 2, <32 x i32> %arg, i32 1, i32 2, i32 3)
  ret <32 x i32> %ret
}

; CHECK: error: <unknown>:0:0: no registers from class available to allocate in function 'no_registers_from_class_available_to_allocate_asm_use'
define void @no_registers_from_class_available_to_allocate_asm_use(<32 x i32> %arg) #0 {
  call void asm sideeffect "; use $0", "v"(<32 x i32> %arg)
  ret void
}

; CHECK: error: <unknown>:0:0: no registers from class available to allocate in function 'no_registers_from_class_available_to_allocate_asm_def'
define <32 x i32> @no_registers_from_class_available_to_allocate_asm_def() #0 {
  %ret = call <32 x i32> asm sideeffect "; def $0", "=v"()
  ret <32 x i32> %ret
}

; CHECK: error: <unknown>:0:0: no registers from class available to allocate in function 'no_registers_from_class_available_to_allocate_undef_asm'
define void @no_registers_from_class_available_to_allocate_undef_asm() #0 {
  call void asm sideeffect "; use $0", "v"(<32 x i32> poison)
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="10,10" }
