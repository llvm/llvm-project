; Check that -kernel-info-end-lto enables kernel-info in the AMD GPU target
; backend.

; REQUIRES: amdgpu-registered-target

; -kernel-info-end-lto inserts kernel-info into LTO pipeline.
; RUN: opt -pass-remarks=kernel-info -disable-output %S/Inputs/test.ll \
; RUN:     -mtriple="amdgcn-amd-amdhsa" \
; RUN:     -passes='lto<O2>' -kernel-info-end-lto 2>&1 | \
; RUN:   FileCheck -match-full-lines %S/Inputs/test.ll

; Omitting -kernel-info-end-lto disables kernel-info.
; RUN: opt -pass-remarks=kernel-info -disable-output %S/Inputs/test.ll \
; RUN:     -mtriple="amdgcn-amd-amdhsa" \
; RUN:     -passes='lto<O2>' 2>&1 | \
; RUN:   FileCheck -allow-empty -check-prefixes=NONE %S/Inputs/test.ll

; Omitting LTO disables kernel-info.
; RUN: opt -pass-remarks=kernel-info -disable-output %S/Inputs/test.ll \
; RUN:     -mtriple="amdgcn-amd-amdhsa" \
; RUN:     -passes='default<O2>' -kernel-info-end-lto 2>&1 | \
; RUN:   FileCheck -allow-empty -check-prefixes=NONE %S/Inputs/test.ll
