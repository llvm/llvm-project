; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

; CHECK: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "foo",{{.*}} retainedNodes: ![[VARS:[0-9]+]]
; CHECK: ![[VARS]] = !{![[PARAM:[0-9]+]], ![[AUTO:[0-9]+]]}
; CHECK: ![[PARAM]] = !DILocalVariable(name: "param", arg: 1, scope: ![[SP]], memorySpace: DW_MSPACE_LLVM_group)
; CHECK: ![[AUTO]]  = !DILocalVariable(name: "auto", scope: ![[SP]], memorySpace: DW_MSPACE_LLVM_private)

!named = !{!0}

!llvm.module.flags = !{!6}
!llvm.dbg.cu = !{!4}

!0 = distinct !DISubprogram(name: "foo", scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !4, retainedNodes: !1)
!1 = !{!2, !3}
!2 = !DILocalVariable(name: "param", arg: 1, scope: !0, memorySpace: DW_MSPACE_LLVM_group)
!3 = !DILocalVariable(name: "auto", scope: !0, memorySpace: DW_MSPACE_LLVM_private)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5)
!5 = !DIFile(filename: "source.c", directory: "/dir")
!6 = !{i32 1, !"Debug Info Version", i32 3}
