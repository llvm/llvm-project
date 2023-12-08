; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK-NOT: error:
!0 = !DISubrange(count: -1)

; CHECK: <stdin>:[[@LINE+1]]:25: error: value for 'count' too small, limit is -1
!0 = !DISubrange(count: -2)
