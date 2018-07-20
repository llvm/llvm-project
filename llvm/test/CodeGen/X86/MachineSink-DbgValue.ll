; RUN: llc < %s | FileCheck %s
; Should sink matching DBG_VALUEs also.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo(i32 %i, i32* nocapture %c) nounwind uwtable readonly ssp !dbg !1 {
  tail call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !6, metadata !DIExpression()), !dbg !12
  %ab = load i32, i32* %c, align 1, !dbg !14
  tail call void @llvm.dbg.value(metadata i32* %c, i64 0, metadata !7, metadata !DIExpression()), !dbg !13
  tail call void @llvm.dbg.value(metadata i32 %ab, i64 0, metadata !10, metadata !DIExpression()), !dbg !14
  %cd = icmp eq i32 %i, 42, !dbg !15
  br i1 %cd, label %bb1, label %bb2, !dbg !15

bb1:                                     ; preds = %0
;CHECK: DEBUG_VALUE: a
;CHECK:      .loc	1 5 5
;CHECK-NEXT: addl
  %gh = add nsw i32 %ab, 2, !dbg !16
  br label %bb2, !dbg !16

bb2:
  %.0 = phi i32 [ %gh, %bb1 ], [ 0, %0 ]
  ret i32 %.0, !dbg !17
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "Apple clang version 3.0 (tags/Apple/clang-211.10.1) (based on LLVM 3.0svn)", isOptimized: true, emissionKind: FullDebug, file: !20, enums: !21, retainedTypes: !21, imports:  null)
!1 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, file: !20, scope: !2, type: !3, retainedNodes: !19)
!2 = !DIFile(filename: "a.c", directory: "/private/tmp")
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "i", line: 2, arg: 1, scope: !1, file: !2, type: !5)
!7 = !DILocalVariable(name: "c", line: 2, arg: 2, scope: !1, file: !2, type: !8)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !0, baseType: !5)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!10 = !DILocalVariable(name: "a", line: 3, scope: !11, file: !2, type: !5)
!11 = distinct !DILexicalBlock(line: 2, column: 25, file: !20, scope: !1)
!12 = !DILocation(line: 2, column: 13, scope: !1)
!13 = !DILocation(line: 2, column: 22, scope: !1)
!14 = !DILocation(line: 3, column: 14, scope: !11)
!15 = !DILocation(line: 4, column: 3, scope: !11)
!16 = !DILocation(line: 5, column: 5, scope: !11)
!17 = !DILocation(line: 7, column: 1, scope: !11)
!19 = !{!6, !7, !10}
!20 = !DIFile(filename: "a.c", directory: "/private/tmp")
!21 = !{}
!22 = !{i32 1, !"Debug Info Version", i32 3}
