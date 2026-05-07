; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:27: error: invalid debug info flag 'DIFlagUnknown'
!0 = !DISubprogram(flags: DIFlagUnknown, type: !2)
!1 = !{null}
!2 = !DISubroutineType(types: !1)
