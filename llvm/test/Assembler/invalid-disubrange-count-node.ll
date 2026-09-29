; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:{{[0-9]+}}: error: 'count' cannot be null
!0 = !DISubrange(count: null)
