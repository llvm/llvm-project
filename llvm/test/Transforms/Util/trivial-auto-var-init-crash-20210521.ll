; RUN: opt -passes=annotation-remarks -pass-remarks-missed=annotation-remarks -disable-output < %s 2>&1 | FileCheck %s

; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "test.ll"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios14.4.0"

%struct.frop = type { ptr }

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memset.p0.i64(ptr noalias nocapture writeonly, i8, i64, i1 immarg) #0

define void @spam() local_unnamed_addr #1 !dbg !3 {
bb:
  call void @llvm.dbg.value(metadata ptr null, metadata !21, metadata !DIExpression()) #3, !dbg !28

; CHECK: remark: :1:0: Call to memset inserted by -ftrivial-auto-var-init. Memory operation size: 0 bytes.
  tail call void @llvm.memset.p0.i64(ptr null, i8 0, i64 0, i1 false), !annotation !33, !dbg !28
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { argmemonly nofree nosync nounwind willreturn }
attributes #1 = { "target-features"="+aes,+crypto,+fp-armv8,+neon,+sha2,+zcm,+zcz" }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!2 = !DIFile(filename: "frop.c", directory: "frop")
!3 = distinct !DISubprogram(scope: !4, file: !4, line: 1, type: !5, scopeLine: 2167, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !19)
!4 = !DIFile(filename: "", directory: "frop")
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_typedef, file: !4, line: 1, baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = distinct !DICompositeType(tag: DW_TAG_union_type, file: !4, line: 1, size: 1664, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, scope: !12, file: !4, line: 1, baseType: !16, size: 192)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !4, line: 1, size: 832, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, scope: !12, file: !4, line: 1, baseType: !15, size: 448)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !4, line: 1, size: 448, elements: !10)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !4, line: 1, size: 192, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, scope: !12, file: !4, line: 1, baseType: !7, size: 64)
!19 = !{!20}
!20 = !DILocalVariable(arg: 1, scope: !3, file: !4, line: 1, type: !7)
!21 = !DILocalVariable(arg: 2, scope: !22, file: !4, line: 1, type: !7)
!22 = distinct !DISubprogram(scope: !4, file: !4, line: 1, type: !23, scopeLine: 1381, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !26)
!23 = !DISubroutineType(types: !24)
!24 = !{!25}
!25 = !DIBasicType(size: 32, encoding: DW_ATE_signed)
!26 = !{!27}
!27 = !DILocalVariable(arg: 1, scope: !22, file: !4, line: 1, type: !7)
!28 = !DILocation(line: 1, scope: !22, inlinedAt: !29)
!29 = distinct !DILocation(line: 1, scope: !30)
!30 = distinct !DILexicalBlock(scope: !31, file: !4, line: 1)
!31 = distinct !DILexicalBlock(scope: !32, file: !4, line: 1)
!32 = distinct !DILexicalBlock(scope: !3, file: !4, line: 1)


!33 = !{ !"auto-init" }
