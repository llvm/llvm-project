; RUN: llvm-as -disable-output <%s 2>&1| FileCheck %s

define void @f() !dbg !14 {
  ret void, !dbg !5
}

!llvm.module.flags = !{!15}
!llvm.dbg.cu = !{!4}

!0 = !{null}
!1 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!2 = !DIFile(filename: "f.c", directory: "/")
!3 = !DISubroutineType(types: !0)
!4 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
; CHECK: !dbg attachment points at wrong subprogram for function
; CHECK: warning: ignoring invalid debug info
!5 = !DILocation(line: 1, scope: !9)
!9 = distinct !DISubprogram(name: "f", scope: !1,
                            file: !2, line: 1, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 2,
                            unit: !4)
!14 = distinct !DISubprogram(name: "f", scope: !1,
                            file: !2, line: 1, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 2,
                            unit: !4)
!15 = !{i32 1, !"Debug Info Version", i32 3}
