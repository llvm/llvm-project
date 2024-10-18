; Check when kernel-info is enabled in the AMD GPU target backend.

; REQUIRES: amdgpu-registered-target

; DEFINE: %{opt} = opt -disable-output %S/Inputs/test.ll \
; DEFINE:              -mtriple="amdgcn-amd-amdhsa" 2>&1
; DEFINE: %{fcheck-on} = FileCheck -match-full-lines %S/Inputs/test.ll
; DEFINE: %{fcheck-off} = FileCheck -allow-empty -check-prefixes=NONE \
; DEFINE:                 %S/Inputs/test.ll

; By default, kernel-info is in the LTO pipeline.  To see output, the LTO
; pipeline must run, -no-kernel-info-end-lto must not be specified, and remarks
; must be enabled.
; RUN: %{opt} -passes='lto<O2>' -pass-remarks=kernel-info | %{fcheck-on}
; RUN: %{opt} -passes='default<O2>' -pass-remarks=kernel-info | %{fcheck-off}
; RUN: %{opt} -passes='lto<O2>' -pass-remarks=kernel-info \
; RUN:        -no-kernel-info-end-lto | %{fcheck-off}
; RUN: %{opt} -passes='lto<O2>' | %{fcheck-off}
