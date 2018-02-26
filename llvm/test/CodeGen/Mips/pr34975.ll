; RUN: llc -mtriple=mips64-unknown-freebsd -target-abi n64 -relocation-model pic -verify-machineinstrs -o /dev/null %s -O2

; Test that the presence of debug information does not cause the branch folder
; to rewrite branches to have negative basic block ids, which would cause the
; long branch pass to crash.

@c = external global i32, align 4

define void @e() !dbg !19 {
entry:
  %0 = load i32, i32* @c, align 4, !dbg !28, !tbaa !31
  %tobool8 = icmp eq i32 %0, 0, !dbg !35
  br i1 %tobool8, label %for.end, label %for.body.preheader, !dbg !35

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !36

for.body:                                         ; preds = %for.body.preheader
  %1 = load i8, i8* undef, align 1, !dbg !36, !tbaa !38
  %conv = zext i8 %1 to i32, !dbg !36
  %cmp = icmp sgt i32 %0, %conv, !dbg !39
  br i1 %cmp, label %if.end, label %if.then, !dbg !40

if.then:                                          ; preds = %for.body
  tail call void @llvm.dbg.value(metadata i32 %conv, metadata !41, metadata !DIExpression()), !dbg !43
  %idxprom5 = zext i8 %1 to i64, !dbg !44
  %call = tail call i32 bitcast (i32 (...)* @g to i32 (i32)*)(i32 signext undef) #3, !dbg !45
  br label %if.end, !dbg !46

if.end:                                           ; preds = %if.then, %for.body
  unreachable

for.end:                                          ; preds = %entry
  ret void
}

declare i32 @g(...)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "/local/scratch/alr48/cheri/llvm/tools/clang/test/CodeGen/<stdin>", directory: "/local/scratch/alr48/cheri/llvm/cmake-build-debug/tools/clang/test/CodeGen")
!2 = !{}
!3 = !{!4, !9, !13, !15}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "a", scope: !0, file: !6, line: 6, type: !7, isLocal: false, isDefinition: true)
!6 = !DIFile(filename: "/crash.c", directory: "/tmp")
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "b", scope: !0, file: !6, line: 7, type: !11, isLocal: false, isDefinition: true)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "c", scope: !0, file: !6, line: 8, type: !12, isLocal: false, isDefinition: true)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "d", scope: !0, file: !6, line: 8, type: !12, isLocal: false, isDefinition: true)
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 7, !"PIC Level", i32 2}
!19 = distinct !DISubprogram(name: "e", scope: !6, file: !6, line: 9, type: !20, isLocal: false, isDefinition: true, scopeLine: 9, isOptimized: true, unit: !0, variables: !22)
!20 = !DISubroutineType(types: !21)
!21 = !{!12}
!22 = !{!23}
!23 = !DILocalVariable(name: "f", scope: !24, file: !6, line: 12, type: !12)
!24 = distinct !DILexicalBlock(scope: !25, file: !6, line: 11, column: 20)
!25 = distinct !DILexicalBlock(scope: !26, file: !6, line: 11, column: 9)
!26 = distinct !DILexicalBlock(scope: !27, file: !6, line: 10, column: 3)
!27 = distinct !DILexicalBlock(scope: !19, file: !6, line: 10, column: 3)
!28 = !DILocation(line: 10, column: 10, scope: !29)
!29 = distinct !DILexicalBlock(scope: !30, file: !6, line: 10, column: 3)
!30 = distinct !DILexicalBlock(scope: !19, file: !6, line: 10, column: 3)
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !DILocation(line: 10, column: 3, scope: !30)
!36 = !DILocation(line: 11, column: 9, scope: !37)
!37 = distinct !DILexicalBlock(scope: !29, file: !6, line: 11, column: 9)
!38 = !{!33, !33, i64 0}
!39 = !DILocation(line: 11, column: 14, scope: !37)
!40 = !DILocation(line: 11, column: 9, scope: !29)
!41 = !DILocalVariable(name: "f", scope: !42, file: !6, line: 12, type: !12)
!42 = distinct !DILexicalBlock(scope: !37, file: !6, line: 11, column: 20)
!43 = !DILocation(line: 12, column: 11, scope: !42)
!44 = !DILocation(line: 13, column: 9, scope: !42)
!45 = !DILocation(line: 13, column: 7, scope: !42)
!46 = !DILocation(line: 14, column: 5, scope: !42)
