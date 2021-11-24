; RUN: llvm-as < %s | llvm-dis - | FileCheck %s

define void @f() !dbg !3 { ret void }

!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!0}

!0 = distinct !DICompileUnit(language: 0, file: !1)
!1 = !DIFile(filename: "-", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 4}
; CHECK: = distinct !DISubprogram({{.*}}retainedNodes: ![[#RETAINED_NODES:]]{{.*}})
!3 = distinct !DISubprogram(name: "f", unit: !0, scope: !0, file: !1, line: 1, retainedNodes: !4)
; CHECK: ![[#RETAINED_NODES]] = !{![[#LIFETIME0:]], ![[#LIFETIME1:]]}
!4 = !{!5, !7}
; CHECK: ![[#LIFETIME0]] = distinct !DILifetime({{.*}}
!5 = distinct !DILifetime(object: !6, location: !DIExpr())
!6 = distinct !DIFragment()
; CHECK: ![[#LIFETIME1]] = distinct !DILifetime({{.*}}
!7 = distinct !DILifetime(object: !6, location: !DIExpr())

; CHECK-NOT: warning: ignoring invalid debug info
