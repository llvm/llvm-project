; RUN: opt < %s -instcombine -S | FileCheck %s

; Make sure to update the debug value after dead code elimination.
; CHECK: %call = call signext i8 @b(i32 6), !dbg !39
; CHECK-NEXT: call void @llvm.dbg.value(metadata i8 %call, metadata !30, metadata !DIExpression(DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_stack_value)), !dbg !38

@e = common local_unnamed_addr global i8 0, align 1, !dbg !0
@c = common local_unnamed_addr global i32 0, align 4, !dbg !6
@d = common local_unnamed_addr global i32 0, align 4, !dbg !10

define signext i8 @b(i32 %f) local_unnamed_addr #0 !dbg !18 {
entry:
  call void @llvm.dbg.value(metadata i32 %f, metadata !22, metadata !DIExpression()), !dbg !23
  %conv = trunc i32 %f to i8, !dbg !24
  ret i8 %conv, !dbg !25
}

define i32 @main() local_unnamed_addr #0 !dbg !26 {
entry:
  %0 = load i8, i8* @e, align 1, !dbg !31, !tbaa !32
  %conv = sext i8 %0 to i32, !dbg !31
  store i32 %conv, i32* @c, align 4, !dbg !35, !tbaa !36
  call void @llvm.dbg.value(metadata i32 -1372423381, metadata !30, metadata !DIExpression()), !dbg !38
  %call = call signext i8 @b(i32 6), !dbg !39
  %conv1 = sext i8 %call to i32, !dbg !39
  call void @llvm.dbg.value(metadata i32 %conv1, metadata !30, metadata !DIExpression()), !dbg !38
  %1 = load i32, i32* @d, align 4, !dbg !40, !tbaa !36
  %call2 = call i32 (...) @optimize_me_not(), !dbg !41
  ret i32 0, !dbg !42
}

declare i32 @optimize_me_not(...) local_unnamed_addr #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "e", scope: !2, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project b306ef12f046353ea5bda4b3b77759e57909a0db)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "a.c", directory: "/Users/davide/llvm/build/bin")
!4 = !{}
!5 = !{!6, !10, !0}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "a", file: !3, line: 1, baseType: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "d", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"PIC Level", i32 2}
!17 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project b306ef12f046353ea5bda4b3b77759e57909a0db)"}
!18 = distinct !DISubprogram(name: "b", scope: !3, file: !3, line: 4, type: !19, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!12, !9}
!21 = !{!22}
!22 = !DILocalVariable(name: "f", arg: 1, scope: !18, file: !3, line: 4, type: !9)
!23 = !DILocation(line: 4, column: 9, scope: !18)
!24 = !DILocation(line: 4, column: 21, scope: !18)
!25 = !DILocation(line: 4, column: 14, scope: !18)
!26 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 5, type: !27, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !29)
!27 = !DISubroutineType(types: !28)
!28 = !{!9}
!29 = !{!30}
!30 = !DILocalVariable(name: "l_1499", scope: !26, file: !3, line: 7, type: !8)
!31 = !DILocation(line: 6, column: 7, scope: !26)
!32 = !{!33, !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !DILocation(line: 6, column: 5, scope: !26)
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !33, i64 0}
!38 = !DILocation(line: 7, column: 5, scope: !26)
!39 = !DILocation(line: 8, column: 12, scope: !26)
!40 = !DILocation(line: 9, column: 11, scope: !26)
!41 = !DILocation(line: 10, column: 3, scope: !26)
!42 = !DILocation(line: 11, column: 1, scope: !26)
