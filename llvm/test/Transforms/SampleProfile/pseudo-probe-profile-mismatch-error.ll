; REQUIRES: x86_64-linux
; RUN: not opt < %S/pseudo-probe-profile-mismatch.ll -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-profile-mismatch.prof  -min-functions-for-staleness-error=1  -precent-mismatch-for-staleness-error=1 -S 2>&1 | FileCheck %s

; CHECK: error: {{.*}}: The input profile significantly mismatches current source code. Please recollect profile to avoid performance regression.
