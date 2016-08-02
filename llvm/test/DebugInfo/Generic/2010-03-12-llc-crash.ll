; RUN: llc -O0 < %s -o /dev/null
; llc should not crash on this optimized out debug info.
; PR6588
declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define void @foo() {
entry:
  call void @llvm.dbg.declare(metadata i32* undef, metadata !0, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  ret void
}

!0 = !DILocalVariable(name: "sy", line: 890, arg: 1, scope: !1, file: !2, type: !7)
!1 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 892, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !8, scope: !3, type: !4)
!2 = !DIFile(filename: "qpainter.h", directory: "QtGui")
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang 1.1", isOptimized: true, emissionKind: 0, file: !9, enums: !10, retainedTypes: !10)
!4 = !DISubroutineType(types: !6)
!5 = !DIFile(filename: "splineeditor.cpp", directory: "src")
!6 = !{null}
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DIFile(filename: "qpainter.h", directory: "QtGui")
!9 = !DIFile(filename: "splineeditor.cpp", directory: "src")
!10 = !{i32 0}
