; Check info on addrspace(0) memory accesses when the target is amdgpu.
;
; The target matters because kernel-info calls
; TargetTransformInfo::getFlatAddressSpace to select addrspace(0).

; REQUIRES: amdgpu-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -mtriple="amdgcn-amd-amdhsa" \
; RUN:     -disable-output %S/Inputs/test.ll 2>&1 | \
; RUN:   FileCheck -match-full-lines -implicit-check-not='addrspace(0)' \
; RUN:       %S/Inputs/test.ll
