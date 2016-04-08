; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; CHECK: .section        .debug_info
; CHECK: DW_AT_const_value
; CHECK-NEXT: 42

define i128 @__foo(i128 %a, i128 %b) nounwind !dbg !3 {
entry:
  tail call void @llvm.dbg.value(metadata i128 42 , i64 0, metadata !1, metadata !DIExpression()), !dbg !11
  %add = add i128 %a, %b, !dbg !11
  ret i128 %add, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!16}

!0 = !{i128 42 }
!1 = !DILocalVariable(name: "MAX", line: 29, scope: !2, file: !4, type: !8)
!2 = distinct !DILexicalBlock(line: 26, column: 0, file: !13, scope: !3)
!3 = distinct !DISubprogram(name: "__foo", linkageName: "__foo", line: 26, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 26, file: !13, scope: !4, type: !6)
!4 = !DIFile(filename: "foo.c", directory: "/tmp")
!5 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "clang", isOptimized: true, emissionKind: FullDebug, file: !13, enums: !15, retainedTypes: !15, subprograms: !12, imports:  null)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "ti_int", line: 78, file: !14, scope: !4, baseType: !10)
!9 = !DIFile(filename: "myint.h", directory: "/tmp")
!10 = !DIBasicType(tag: DW_TAG_base_type, size: 128, align: 128, encoding: DW_ATE_signed)
!11 = !DILocation(line: 29, scope: !2)
!12 = !{!3}
!13 = !DIFile(filename: "foo.c", directory: "/tmp")
!14 = !DIFile(filename: "myint.h", directory: "/tmp")
!15 = !{}
!16 = !{i32 1, !"Debug Info Version", i32 3}
