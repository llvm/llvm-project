; COM: Test that the default behavior persists (the llc-specific handler prints all errors).
; RUN: not llc -mtriple=amdgcn -verify-machineinstrs=0 -global-isel=false < %s 2>&1 | FileCheck -check-prefix=ALL-ERRORS %s
; COM: Do not halt on the first error when the llc-specific handler is not loaded.
; RUN: not llc -mtriple=amdgcn -verify-machineinstrs=0 -global-isel=false -no-diag-handler -halt-on-first-diag-error=none < %s 2>&1 | FileCheck -check-prefix=ALL-ERRORS %s

; COM: Now halt on the first error by disabling the llc-specific handler and test the different halt actions
; RUN: not llc -mtriple=amdgcn -verify-machineinstrs=0 -global-isel=false -no-diag-handler -halt-on-first-diag-error=exit < %s 2>&1 | FileCheck -check-prefix=FIRST-ERROR %s
; COM: Same error message as in -halt-on-first-diag-error=exit but with a crash.
; RUN: not --crash llc -mtriple=amdgcn -verify-machineinstrs=0 -global-isel=false -no-diag-handler -halt-on-first-diag-error=abort < %s 2>&1 | FileCheck -check-prefix=FIRST-ERROR %s

; ALL-ERRORS: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_i32 void (): illegal VGPR to SGPR copy
; FIRST-ERROR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_i32 void (): illegal VGPR to SGPR copy
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_i32() #0 {
  %vgpr = call i32 asm sideeffect "; def $0", "=${v1}"()
  call void asm sideeffect "; use $0", "${s9}"(i32 %vgpr)
  ret void
}

; ALL-ERRORS: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v2i32 void (): illegal VGPR to SGPR copy
; FIRST-ERROR-NOT: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v2i32 void (): illegal VGPR to SGPR copy
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v2i32() #0 {
  %vgpr = call <2 x i32> asm sideeffect "; def $0", "=${v[0:1]}"()
  call void asm sideeffect "; use $0", "${s[10:11]}"(<2 x i32> %vgpr)
  ret void
}
