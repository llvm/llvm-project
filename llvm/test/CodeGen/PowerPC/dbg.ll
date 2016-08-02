; RUN: llc < %s -break-anti-dependencies=all -march=ppc64 -mcpu=g5 | FileCheck %s
; CHECK-LABEL: main:

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind readnone !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !15, metadata !DIExpression()), !dbg !17
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !16, metadata !DIExpression()), !dbg !18
  %add = add nsw i32 %argc, 1, !dbg !19
  ret i32 %add, !dbg !19
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1", isOptimized: true, emissionKind: 0, file: !21, enums: !1, retainedTypes: !1, subprograms: !3, globals: !1, imports: !1)
!1 = !{}
!3 = !{!5}
!5 = distinct !DISubprogram(name: "main", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !21, scope: null, type: !7, variables: !13)
!6 = !DIFile(filename: "dbg.c", directory: "/src")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9, !10}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!13 = !{!15, !16}
!15 = !DILocalVariable(name: "argc", line: 1, arg: 1, scope: !5, file: !6, type: !9)
!16 = !DILocalVariable(name: "argv", line: 1, arg: 2, scope: !5, file: !6, type: !10)
!17 = !DILocation(line: 1, column: 14, scope: !5)
!18 = !DILocation(line: 1, column: 26, scope: !5)
!19 = !DILocation(line: 2, column: 3, scope: !20)
!20 = distinct !DILexicalBlock(line: 1, column: 34, file: !21, scope: !5)
!21 = !DIFile(filename: "dbg.c", directory: "/src")
!22 = !{i32 1, !"Debug Info Version", i32 3}
