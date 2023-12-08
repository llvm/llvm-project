; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!0 = !{}

; CHECK-NOT: error:
!1 = !DILocation(line: 4294967295, scope: !0)

; CHECK: <stdin>:[[@LINE+1]]:24: error: value for 'line' too large, limit is 4294967295
!2 = !DILocation(line: 4294967296, scope: !0)
