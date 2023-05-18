; RUN: llvm-as -disable-output <%s 2>&1| FileCheck %s

declare !dbg !12 i32 @declared_only()

!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!5}

!2 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !6, producer: "clang", emissionKind: FullDebug)
!6 = !DIFile(filename: "a.cpp", directory: "/")
!7 = !{}
!11 = !DISubroutineType(types: !7)

!12 = !DISubprogram(name: "declared_only", scope: !6, file: !6, line: 2, type: !11, spFlags: DISPFlagOptimized, retainedNodes: !7, declaration: !13)
!13 = !DISubprogram(name: "declared_only", scope: !6, file: !6, line: 2, type: !11, spFlags: DISPFlagOptimized, retainedNodes: !7)
; CHECK: subprogram declaration must not have a declaration field
; CHECK: warning: ignoring invalid debug info
