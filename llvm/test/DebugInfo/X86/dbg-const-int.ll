; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %s \
; RUN:    | llvm-dwarfdump -debug-dump=info - | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"
; Radar 9511391

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_const_value [DW_FORM_sdata]   (42)
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "i"

define i32 @foo() nounwind uwtable readnone optsize ssp !dbg !1 {
entry:
  tail call void @llvm.dbg.value(metadata i32 42, i64 0, metadata !6, metadata !DIExpression()), !dbg !9
  ret i32 42, !dbg !10
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 132191)", isOptimized: true, emissionKind: FullDebug, file: !13, enums: !14, retainedTypes: !14, subprograms: !11, imports:  null)
!1 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !13, scope: !2, type: !3, variables: !12)
!2 = !DIFile(filename: "a.c", directory: "/private/tmp")
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "i", line: 2, scope: !7, file: !2, type: !5)
!7 = distinct !DILexicalBlock(line: 1, column: 11, file: !13, scope: !1)
!8 = !{i32 42}
!9 = !DILocation(line: 2, column: 12, scope: !7)
!10 = !DILocation(line: 3, column: 2, scope: !7)
!11 = !{!1}
!12 = !{!6}
!13 = !DIFile(filename: "a.c", directory: "/private/tmp")
!14 = !{}
!15 = !{i32 1, !"Debug Info Version", i32 3}
