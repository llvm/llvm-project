; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s
; Check common error from old format.

; CHECK: {{.*}}:[[@LINE+1]]:{{[0-9]+}}: error: unexpected type in metadata definition
!0 = metadata !{}
