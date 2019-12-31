; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Force a specific numbering.
; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
; CHECK: !llvm.dbg.cu = !{!8, !9, !10}
!llvm.dbg.cu = !{!8, !9, !10}

!0 = distinct !{}
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = distinct !{}
!3 = distinct !{}
!4 = distinct !{}
!5 = distinct !{}
!6 = distinct !{}
!7 = distinct !{}

; CHECK: !8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, flags: "-O2", runtimeVersion: 2, splitDebugFilename: "abc.debug", emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !5, imports: !6, macros: !7, dwoId: 42)
!8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang",
                             isOptimized: true, flags: "-O2", runtimeVersion: 2,
                             splitDebugFilename: "abc.debug",
                             emissionKind: FullDebug,
                             enums: !2, retainedTypes: !3,
                             globals: !5, imports: !6, macros: !7, dwoId: 42, splitDebugInlining: true)

; CHECK: !9 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!9 = distinct !DICompileUnit(language: 12, file: !1, producer: "",
                             isOptimized: false, flags: "", runtimeVersion: 0,
                             splitDebugFilename: "", emissionKind: NoDebug)

; CHECK: !10 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, flags: "-O2", runtimeVersion: 2, splitDebugFilename: "abc.debug", emissionKind: LineTablesOnly, splitDebugInlining: false)
!10 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang",
                             isOptimized: true, flags: "-O2", runtimeVersion: 2,
                             splitDebugFilename: "abc.debug",
                             emissionKind: LineTablesOnly, splitDebugInlining: false)

!llvm.module.flags = !{!11}
!11 = !{i32 2, !"Debug Info Version", i32 3}
