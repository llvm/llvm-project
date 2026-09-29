; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:41: error: value for 'lowerBound' too large, limit is 9223372036854775807
!0 = !DISubrange(count: 30, lowerBound: 9223372036854775808)
