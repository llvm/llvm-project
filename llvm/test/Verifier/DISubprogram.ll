; RUN: llvm-as -disable-output <%s 2>&1| FileCheck %s

define void @f() !dbg !14 {
  ret void
}

!0 = !{null}
!1 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !DISubroutineType(types: !0)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!8 = distinct !DICompileUnit(language: DW_LANG_Swift, producer: "clang",
                             file: !2, emissionKind: 2)
; CHECK: invalid thrown type
; CHECK: warning: ignoring invalid debug info
!13 = !{!14}
!14 = distinct !DISubprogram(name: "f", scope: !1,
                            file: !2, line: 1, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 2,
                            unit: !8, thrownTypes: !13)
!15 = !{i32 1, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!15}
!llvm.dbg.cu = !{!8}
