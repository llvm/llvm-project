; Check info on flat address space memory accesses when the target is nvptx.
;
; The target matters because kernel-info calls
; TargetTransformInfo::getFlatAddressSpace to select the flat address space.

; REQUIRES: nvptx-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -mtriple="nvptx64-nvidia-cuda" \
; RUN:     -disable-output %S/Inputs/test.ll 2>&1 | \
; RUN:   FileCheck -match-full-lines -implicit-check-not='flat address space' \
; RUN:       %S/Inputs/test.ll
