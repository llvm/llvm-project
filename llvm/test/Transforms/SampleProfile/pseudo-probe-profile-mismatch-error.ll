; REQUIRES: x86_64-linux
; RUN: opt < %S/pseudo-probe-profile-mismatch.ll -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-profile-mismatch.prof -checksum-mismatch-func-hot-block-skip=0 -checksum-mismatch-num-func-skip=1 -checksum-mismatch-error-threshold=1 -S 2>%t -o %t.ll
; RUN: FileCheck %s --input-file %t

; CHECK: error: [[*]]: The FDO profile is too old and will cause big performance regression, please drop the profile and collect a new one.
