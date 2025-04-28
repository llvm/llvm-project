; RUN: llvm-as < %s | llvm-dis - | FileCheck %s

!llvm.module.flags = !{!0, !1}

; CHECK: !llvm.dbg.retainedNodes = !{![[#LIFETIME0:]], ![[#LIFETIME1:]]}
!llvm.dbg.retainedNodes = !{!2, !4}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 4}

; CHECK: ![[#LIFETIME0]] = distinct !DILifetime({{.*}}
!2 = distinct !DILifetime(object: !3, location: !DIExpr())
!3 = distinct !DIFragment()
; CHECK: ![[#LIFETIME1]] = distinct !DILifetime({{.*}}
!4 = distinct !DILifetime(object: !3, location: !DIExpr())

; CHECK-NOT: warning: ignoring invalid debug info in <stdin>
