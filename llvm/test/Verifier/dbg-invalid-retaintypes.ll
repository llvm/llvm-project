; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: warning: ignoring invalid debug info

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 131941)", isOptimized: true, emissionKind: FullDebug, file: !2, retainedTypes: !1)
!1 = distinct !DISubprogram(name: "main", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, scope: !2)
!2 = !DIFile(filename: "/davide/test", directory: "/")
!3 = !{i32 1, !"Debug Info Version", i32 3}
