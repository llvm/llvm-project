; Bitcode originally compiled with 3.7_rc2.  3.7 had redundant (but
; mandatory) tag fields on DILocalVariable.
;
; RUN: llvm-dis < %s.bc -o - | llvm-as | llvm-dis | FileCheck %s

; CHECK: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "foo",{{.*}} type: ![[TYPE:[0-9]+]],{{.*}} retainedNodes: ![[VARS:[0-9]+]]
; CHECK: ![[TYPE]] = !DISubroutineType(types: ![[ARGS:[0-9]+]])
; CHECK: ![[ARGS]] = !{null}
; CHECK: ![[VARS]] = !{![[PARAM:[0-9]+]], ![[AUTO:[0-9]+]]}
; CHECK: ![[PARAM]] = !DILocalVariable(name: "param", arg: 1, scope: ![[SP]])
; CHECK: ![[AUTO]]  = !DILocalVariable(name: "auto", scope: ![[SP]])

!named = !{!0}

!llvm.module.flags = !{!6}
!llvm.dbg.cu = !{!4}

!0 = distinct !DISubprogram(name: "foo", scope: null, type: !7, isLocal: false, isDefinition: true, isOptimized: false, unit: !4, retainedNodes: !1)
!1 = !{!2, !3}
!2 = !DILocalVariable(name: "param", arg: 1, scope: !0)
!3 = !DILocalVariable(name: "auto", scope: !0)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5)
!5 = !DIFile(filename: "source.c", directory: "/dir")
!6 = !{i32 1, !"Debug Info Version", i32 3}
!7 = !DISubroutineType(types: !8)
!8 = !{null}
