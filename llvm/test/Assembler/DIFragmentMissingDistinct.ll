; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: [[# @LINE + 1]]:6: error: missing 'distinct', required for !DIFragment
!0 = !DIFragment()
