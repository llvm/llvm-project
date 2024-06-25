; This test checks that we do not accidently mutate the debug info when
; inserting loop parallel metadata.
; RUN: opt %loadNPMPolly < %s  -S -polly -passes=polly-codegen -polly-ast-detect-parallel | FileCheck %s
; CHECK-NOT: !7 = !{!7}
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = common global ptr null, align 8

; Function Attrs: nounwind uwtable
define void @foo() !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !9, metadata !19), !dbg !20
  %0 = load ptr, ptr @A, align 8, !dbg !21, !tbaa !23
  br label %for.body, !dbg !27

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %0, i64 %indvars.iv, !dbg !21
  %1 = load i32, ptr %arrayidx, align 4, !dbg !21, !tbaa !30
  %add = add nsw i32 %1, 1, !dbg !21
  store i32 %add, ptr %arrayidx, align 4, !dbg !21, !tbaa !30
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !27
  %exitcond = icmp eq i64 %indvars.iv, 1, !dbg !27
  br i1 %exitcond, label %for.end, label %for.body, !dbg !27

for.end:                                          ; preds = %for.body
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, globals: !12, imports: !2)
!1 = !DIFile(filename: "t2.c", directory: "/local/mnt/workspace/build/tip-Release")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: true, isOptimized: true, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !8)
!5 = !DIFile(filename: "t2.c", directory: "/local/mnt/workspace/build/tip-Release")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9}
!9 = !DILocalVariable(name: "i", line: 4, scope: !10, file: !5, type: !11)
!10 = distinct !DILexicalBlock(line: 4, column: 3, file: !1, scope: !4)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "A", line: 2, isLocal: false, isDefinition: true, scope: null, file: !5, type: !14), expr: !DIExpression())
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{!"clang version 3.6.0 "}
!18 = !{i32 0}
!19 = !DIExpression()
!20 = !DILocation(line: 4, column: 12, scope: !10)
!21 = !DILocation(line: 5, column: 5, scope: !22)
!22 = distinct !DILexicalBlock(line: 4, column: 3, file: !1, scope: !10)
!23 = !{!24, !24, i64 0}
!24 = !{!"any pointer", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 4, column: 3, scope: !28)
!28 = !DILexicalBlockFile(discriminator: 2, file: !1, scope: !29)
!29 = !DILexicalBlockFile(discriminator: 1, file: !1, scope: !22)
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !25, i64 0}
!32 = !DILocation(line: 6, column: 1, scope: !4)
