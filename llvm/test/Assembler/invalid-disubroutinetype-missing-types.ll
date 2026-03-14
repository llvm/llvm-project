; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:65: error: missing required field 'types'
!29 = !DISubroutineType(flags: DIFlagPublic | DIFlagStaticMember)
