; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -o - %s 2>&1 | FileCheck -check-prefix=SDAG-ERR %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -o - %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s

; SDAG-ERR: LLVM ERROR: unsupported libcall legalization
; GISEL-ERR: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(s128) = G_SDIV %{{[0-9]+}}:_, %{{[0-9]+}}:_ (in function: v_sdiv_i128_vv)
define i128 @v_sdiv_i128_vv(i128 %lhs, i128 %rhs) {
  %shl = sdiv i128 %lhs, %rhs
  ret i128 %shl
}
