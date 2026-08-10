; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s
; Check common error from old format.

define void @foo() {
; CHECK: {{.*}}:[[@LINE+1]]:{{[0-9]+}}: error: invalid metadata-value-metadata roundtrip
  ret void, !bar !{metadata !0}
}
!0 = !{}
