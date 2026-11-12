; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:41: error: value for 'lowerBound' too small, limit is -9223372036854775808
!0 = !DISubrange(count: 30, lowerBound: -9223372036854775809)
