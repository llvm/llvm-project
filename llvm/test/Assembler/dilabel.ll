; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: !named = !{!0, ![[FOO:[0-9]+]], ![[BAR:[0-9]+]]}
!named = !{!0, !1, !2}

!llvm.module.flags = !{!3}
!llvm.dbg.cu = !{!4}

; CHECK-DAG: ![[FOO]] = !DILabel(scope: !0, name: "foo", file: ![[FILE:[0-9]+]], line: 7)
; CHECK-DAG: ![[BAR]] = !DILabel(scope: !0, name: "bar", file: ![[FILE]], line: 0)
!0 = distinct !DISubprogram(unit: !4, type: !5)
!1 = !DILabel(scope: !0, name: "foo", file: !6, line: 7)
!2 = !DILabel(scope: !0, name: "bar", file: !6, line: 0)

!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !6,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!5 = !DISubroutineType(types: !7)
!6 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!7 = !{null}
