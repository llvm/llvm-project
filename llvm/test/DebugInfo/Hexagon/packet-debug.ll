; Passes if line number 9 is generated in the line table

; RUN: llc %s -mtriple=hexagon-unknown-elf -filetype=obj -mcpu=hexagonv69 -o - | llvm-dwarfdump --debug-line - | FileCheck %s

; CHECK: 9 9 1 0 0 0 is_stmt prologue_end

@glob = common dso_local local_unnamed_addr global i32 0, align 4, !dbg !0
@.str = private unnamed_addr constant [23 x i8] c"Factorial of %d is %d\0A\00", align 1, !dbg !5
define dso_local i32 @factorial(i32 %i) local_unnamed_addr !dbg !18 {
entry:
    #dbg_value(i32 %i, !22, !DIExpression(), !24)
  %cmp = icmp eq i32 %i, 1, !dbg !25
  br i1 %cmp, label %common.ret, label %if.end, !dbg !25
common.ret:                                       ; preds = %entry, %if.end
  %common.ret.op = phi i32 [ %mul, %if.end ], [ 1, %entry ]
  ret i32 %common.ret.op, !dbg !27
if.end:                                           ; preds = %entry
  %sub = add nsw i32 %i, -1, !dbg !28
  %call = tail call i32 @factorial(i32 noundef %sub), !dbg !29
  %mul = mul nsw i32 %call, %i, !dbg !30
    #dbg_value(i32 %mul, !23, !DIExpression(), !24)
  store i32 %mul, ptr @glob, align 4, !dbg !31, !tbaa !32
  br label %common.ret, !dbg !27
}
define dso_local noundef i32 @main(i32 noundef %argc, ptr noundef readnone captures(none) %argv) local_unnamed_addr !dbg !36 {
entry:
    #dbg_value(i32 %argc, !42, !DIExpression(), !46)
    #dbg_value(ptr %argv, !43, !DIExpression(), !46)
    #dbg_value(i32 10, !44, !DIExpression(), !46)
  %call = tail call i32 @factorial(i32 noundef 10), !dbg !47
  %call1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 10, i32 noundef %call), !dbg !48
  ret i32 0, !dbg !50
}
declare !dbg !51 dso_local noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15, !16}
!llvm.ident = !{!17}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glob", scope: !2, file: !3, line: 4, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "QuIC LLVM Hexagon Clang version 21.0 Engineering Release: hexagon-clang-210", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "fact.c", directory: ".")
!4 = !{!5, !0}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(scope: null, file: !3, line: 38, type: !7, isLocal: true, isDefinition: true)
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 184, elements: !9)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!9 = !{!10}
!10 = !DISubrange(count: 23)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 7, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!17 = !{!"QuIC LLVM Hexagon Clang version 21.0 Engineering Release: hexagon-clang-210"}
!18 = distinct !DISubprogram(name: "factorial", scope: !3, file: !3, line: 6, type: !19, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!11, !11}
!21 = !{!22, !23}
!22 = !DILocalVariable(name: "i", arg: 1, scope: !18, file: !3, line: 6, type: !11)
!23 = !DILocalVariable(name: "j", scope: !18, file: !3, line: 8, type: !11)
!24 = !DILocation(line: 0, scope: !18)
!25 = !DILocation(line: 9, column: 9, scope: !26)
!26 = distinct !DILexicalBlock(scope: !18, file: !3, line: 9, column: 7)
!27 = !DILocation(line: 14, column: 1, scope: !18)
!28 = !DILocation(line: 11, column: 23, scope: !18)
!29 = !DILocation(line: 11, column: 11, scope: !18)
!30 = !DILocation(line: 11, column: 9, scope: !18)
!31 = !DILocation(line: 12, column: 8, scope: !18)
!32 = !{!33, !33, i64 0}
!33 = !{!"int", !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 16, type: !37, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !41)
!37 = !DISubroutineType(types: !38)
!38 = !{!11, !11, !39}
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !40, size: 32)
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 32)
!41 = !{!42, !43, !44}
!42 = !DILocalVariable(name: "argc", arg: 1, scope: !36, file: !3, line: 16, type: !11)
!43 = !DILocalVariable(name: "argv", arg: 2, scope: !36, file: !3, line: 16, type: !39)
!44 = !DILocalVariable(name: "base", scope: !36, file: !3, line: 18, type: !45)
!45 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!46 = !DILocation(line: 0, scope: !36)
!47 = !DILocation(line: 38, column: 43, scope: !36)
!48 = !DILocation(line: 38, column: 3, scope: !49)
!49 = !DILexicalBlockFile(scope: !36, file: !3, discriminator: 2)
!50 = !DILocation(line: 39, column: 3, scope: !36)
!51 = !DISubprogram(name: "printf", scope: !52, file: !52, line: 160, type: !53, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!52 = !DIFile(filename: "stdio.h", directory: ".")
!53 = !DISubroutineType(types: !54)
!54 = !{!11, !55, null}
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !56, size: 32)
!56 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
