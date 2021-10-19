; RUN: opt < %s -loop-reduce -S | FileCheck %s
;
; Test that LSR avoids crashing on very large integer inputs. It should
; discard the variable location by creating an undef dbg.value.
;
; CHECK: call void @llvm.dbg.value(metadata i128 undef,

source_filename = "<stdin>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @qux() local_unnamed_addr !dbg !10 {
entry:
  br label %for.body, !dbg !17

for.body:                                         ; preds = %for.body.for.body_crit_edge, %entry
  %0 = phi i128 [ 0, %entry ], [ %.pre8, %for.body.for.body_crit_edge ], !dbg !19
  %add6.i.i = add i128 %0, 18446744073709551615, !dbg !35
  call void @llvm.dbg.value(metadata i128 %add6.i.i, metadata !25, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 128)), !dbg !36
  br label %for.body.for.body_crit_edge

for.body.for.body_crit_edge:                      ; preds = %for.body
  %.pre8 = load i128, i128* undef, align 16, !dbg !19, !tbaa !37
  br label %for.body, !dbg !17
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/beans.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !5, line: 1, baseType: !6)
!5 = !DIFile(filename: "/tmp/beans.c", directory: "")
!6 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = distinct !DISubprogram(name: "qux", scope: !5, file: !5, line: 27, type: !11, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "croix", file: !5, line: 2, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint128_t", file: !1, baseType: !16)
!16 = !DIBasicType(name: "unsigned __int128", size: 128, encoding: DW_ATE_unsigned)
!17 = !DILocation(line: 30, column: 3, scope: !18)
!18 = distinct !DILexicalBlock(scope: !10, file: !5, line: 30, column: 3)
!19 = !DILocation(line: 12, column: 25, scope: !20, inlinedAt: !30)
!20 = distinct !DISubprogram(name: "bar", scope: !5, file: !5, line: 8, type: !21, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !24)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !23, !13}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!24 = !{!25}
!25 = !DILocalVariable(name: "tmp", scope: !20, file: !5, line: 9, type: !26)
!26 = !DIDerivedType(tag: DW_TAG_typedef, name: "xyzzy", file: !5, line: 3, baseType: !27)
!27 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 512, elements: !28)
!28 = !{!29}
!29 = !DISubrange(count: 4)
!30 = distinct !DILocation(line: 23, column: 3, scope: !31, inlinedAt: !32)
!31 = distinct !DISubprogram(name: "baz", scope: !5, file: !5, line: 21, type: !11, scopeLine: 21, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!32 = distinct !DILocation(line: 31, column: 5, scope: !33)
!33 = distinct !DILexicalBlock(scope: !34, file: !5, line: 30, column: 31)
!34 = distinct !DILexicalBlock(scope: !18, file: !5, line: 30, column: 3)
!35 = !DILocation(line: 12, column: 23, scope: !20, inlinedAt: !30)
!36 = !DILocation(line: 0, scope: !20, inlinedAt: !30)
!37 = !{!38, !38, i64 0}
!38 = !{!"__int128", !39, i64 0}
!39 = !{!"omnipotent char", !40, i64 0}
!40 = !{!"Simple C/C++ TBAA"}
