; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; CHECK: Compile unit: DW_LANG_C99 from /tmp/test1.c
; CHECK: Compile unit: DW_LNAME_C from /tmp/test2.c
; CHECK: Compile unit: unknown-language(0) from /tmp/test3.c

!llvm.dbg.cu = !{!0, !6, !10}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "test1.c", directory: "/tmp")
!2 = !{}
!3 = !DIFile(filename: "test1.c", directory: "/tmp")
!4 = !DISubroutineType(types: !7)
!5 = !{null}
!6 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_C, producer: "clang", isOptimized: false, emissionKind: FullDebug, file: !7, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!7 = !DIFile(filename: "test2.c", directory: "/tmp")
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = distinct !DICompileUnit(sourceLanguageName: 0, producer: "clang", isOptimized: false, emissionKind: FullDebug, file: !11, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!11 = !DIFile(filename: "test3.c", directory: "/tmp")
