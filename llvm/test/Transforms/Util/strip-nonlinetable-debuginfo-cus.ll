; RUN: opt -S -strip-nonlinetable-debuginfo %s -o - | FileCheck %s
!llvm.dbg.cu = !{!2, !6}
!llvm.gcov = !{!3}
!llvm.module.flags = !{!7}

!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
; The first CU is used for the line table, the second one is a module skeleton
; and should be stripped.
; CHECK: !llvm.dbg.cu = !{![[CU:[0-9]+]]}
; CHECK: ![[CU]] = distinct !DICompileUnit({{.*}}"abc.debug"{{.*}}LineTablesOnly
; CHECK-NOT: retainedTypes:
; CHECK-SAME: )
; CHECK-NOT: DICompositeType
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang",
                             isOptimized: true, flags: "-O2", runtimeVersion: 2,
                             splitDebugFilename: "abc.debug", emissionKind: FullDebug,
                             retainedTypes: !4)
!3 = !{!"path/to/file.o", !2}
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_structure_type, file: !1, identifier: "ThisWillBeStripped")
!6 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang",
                             splitDebugFilename: "abc.dwo", emissionKind: FullDebug,
                             dwoId: 1234)
!7 = !{i32 1, !"Debug Info Version", i32 3}
