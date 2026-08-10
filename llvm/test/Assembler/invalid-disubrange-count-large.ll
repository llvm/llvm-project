; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK-NOT: error:
!0 = !DISubrange(count: 9223372036854775807)

; CHECK: <stdin>:[[@LINE+1]]:25: error: value for 'count' too large, limit is 9223372036854775807
!1 = !DISubrange(count: 9223372036854775808)
