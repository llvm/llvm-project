; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!0 = !{}

; CHECK: <stdin>:[[@LINE+1]]:38: error: field 'line' cannot be specified more than once
!1 = !DILocation(line: 3, scope: !0, line: 3)
