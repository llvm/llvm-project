; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu10.10 < %s 2>&1 | FileCheck -check-prefix=SDAG-ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu10.10 < %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu11.00 < %s 2>&1 | FileCheck -check-prefix=SDAG-ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu11.00 < %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s

; v_mad_f16 and v_mac_f16 only exist on GFX8 and GFX9, so
; llvm.amdgcn.fmad.ftz.f16 cannot be selected on GFX10+. This used to select
; a pseudo instruction with no encoding and crash later in the pipeline.

; SDAG-ERR: LLVM ERROR: Cannot select: {{.*}} = AMDGPUISD::FMAD_FTZ
; GISEL-ERR: LLVM ERROR: cannot select: {{.*}} = G_INTRINSIC intrinsic(@llvm.amdgcn.fmad.ftz)

define half @fmad_ftz_f16(half %a, half %b, half %c) {
  %r = call half @llvm.amdgcn.fmad.ftz.f16(half %a, half %b, half %c)
  ret half %r
}
