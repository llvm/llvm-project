; REQUIRES: asserts
; RUN: llc -debug-only=isel %s -o /dev/null 2>&1 | FileCheck --match-full-lines %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@str = private unnamed_addr constant [2 x i8] c"1\00", align 1
@str.12 = private unnamed_addr constant [2 x i8] c"2\00", align 1
@str.13 = private unnamed_addr constant [2 x i8] c"3\00", align 1
@str.14 = private unnamed_addr constant [2 x i8] c"4\00", align 1
@str.15 = private unnamed_addr constant [2 x i8] c"5\00", align 1
@str.16 = private unnamed_addr constant [2 x i8] c"6\00", align 1
@str.17 = private unnamed_addr constant [2 x i8] c"7\00", align 1
@str.18 = private unnamed_addr constant [2 x i8] c"8\00", align 1
@str.19 = private unnamed_addr constant [2 x i8] c"9\00", align 1
@str.20 = private unnamed_addr constant [3 x i8] c"10\00", align 1
@str.21 = private unnamed_addr constant [3 x i8] c"11\00", align 1
@str.22 = private unnamed_addr constant [3 x i8] c"12\00", align 1



; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(i32 noundef %cond) local_unnamed_addr #0 !dbg !42 {
;CHECK: Initial selection DAG: %bb.{{[0-9]+}} 'foo:entry'
;CHECK: SelectionDAG has 5 nodes:
;CHECK:     [[TMP1:t.*]]: ch,glue = EntryToken
;CHECK:   [[TMP2:t.*]]: i64,ch = CopyFromReg [[TMP1]], Register:i64 %{{[0-9]+}}, jump_table.c:4:3
;CHECK:   t{{[0-9]+}}: ch = br_jt [[TMP2]]:1, JumpTable:i64<0>, [[TMP2]], jump_table.c:4:3

entry:
  call void @llvm.dbg.value(metadata i32 %cond, metadata !47, metadata !DIExpression()), !dbg !48
  switch i32 %cond, label %sw.epilog [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb3
    i32 4, label %sw.bb5
    i32 5, label %sw.bb7
    i32 6, label %sw.bb9
    i32 7, label %sw.bb11
    i32 8, label %sw.bb13
    i32 9, label %sw.bb15
    i32 10, label %sw.bb17
    i32 11, label %sw.bb19
    i32 12, label %sw.bb21
  ], !dbg !49

sw.bb:                                            ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str), !dbg !50
  br label %sw.bb1, !dbg !50

sw.bb1:                                           ; preds = %entry, %sw.bb
  %puts23 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.12), !dbg !52
  br label %sw.bb3, !dbg !52

sw.bb3:                                           ; preds = %entry, %sw.bb1
  %puts24 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.13), !dbg !53
  br label %sw.bb5, !dbg !53

sw.bb5:                                           ; preds = %entry, %sw.bb3
  %puts25 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14), !dbg !54
  br label %sw.bb7, !dbg !54

sw.bb7:                                           ; preds = %entry, %sw.bb5
  %puts26 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15), !dbg !55
  br label %sw.bb9, !dbg !55

sw.bb9:                                           ; preds = %entry, %sw.bb7
  %puts27 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16), !dbg !56
  br label %sw.bb11, !dbg !56

sw.bb11:                                          ; preds = %entry, %sw.bb9
  %puts28 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17), !dbg !57
  br label %sw.bb13, !dbg !57

sw.bb13:                                          ; preds = %entry, %sw.bb11
  %puts29 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.18), !dbg !58
  br label %sw.bb15, !dbg !58

sw.bb15:                                          ; preds = %entry, %sw.bb13
  %puts30 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.19), !dbg !59
  br label %sw.bb17, !dbg !59

sw.bb17:                                          ; preds = %entry, %sw.bb15
  %puts31 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.20), !dbg !60
  br label %sw.bb19, !dbg !60

sw.bb19:                                          ; preds = %entry, %sw.bb17
  %puts32 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.21), !dbg !61
  br label %sw.bb21, !dbg !61

sw.bb21:                                          ; preds = %entry, %sw.bb19
  %puts33 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.22), !dbg !62
  ret void, !dbg !63

sw.epilog:                                        ; preds = %entry
  ret void, !dbg !63
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!34, !35, !36, !37, !38, !39, !40}
!llvm.ident = !{!41}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (https://github.com/llvm/llvm-project.git 24cf476bd6d144b9fa28325b4d16e3c9dbfc4d4f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "jump_table.c", directory: "/export/compilers/llvm-project", checksumkind: CSK_MD5, checksum: "0847b70de02e07499cd0177d1bdc6dae")
!2 = !{!3, !9, !11, !13, !15, !17, !19, !21, !23, !25, !30, !32}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(scope: null, file: !1, line: 6, type: !5, isLocal: true, isDefinition: true)
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 24, elements: !7)
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!7 = !{!8}
!8 = !DISubrange(count: 3)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(scope: null, file: !1, line: 8, type: !5, isLocal: true, isDefinition: true)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(scope: null, file: !1, line: 10, type: !5, isLocal: true, isDefinition: true)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(scope: null, file: !1, line: 12, type: !5, isLocal: true, isDefinition: true)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(scope: null, file: !1, line: 14, type: !5, isLocal: true, isDefinition: true)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(scope: null, file: !1, line: 16, type: !5, isLocal: true, isDefinition: true)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(scope: null, file: !1, line: 18, type: !5, isLocal: true, isDefinition: true)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(scope: null, file: !1, line: 20, type: !5, isLocal: true, isDefinition: true)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(scope: null, file: !1, line: 22, type: !5, isLocal: true, isDefinition: true)
!25 = !DIGlobalVariableExpression(var: !26, expr: !DIExpression())
!26 = distinct !DIGlobalVariable(scope: null, file: !1, line: 24, type: !27, isLocal: true, isDefinition: true)
!27 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 32, elements: !28)
!28 = !{!29}
!29 = !DISubrange(count: 4)
!30 = !DIGlobalVariableExpression(var: !31, expr: !DIExpression())
!31 = distinct !DIGlobalVariable(scope: null, file: !1, line: 26, type: !27, isLocal: true, isDefinition: true)
!32 = !DIGlobalVariableExpression(var: !33, expr: !DIExpression())
!33 = distinct !DIGlobalVariable(scope: null, file: !1, line: 28, type: !27, isLocal: true, isDefinition: true)
!34 = !{i32 7, !"Dwarf Version", i32 5}
!35 = !{i32 2, !"Debug Info Version", i32 3}
!36 = !{i32 1, !"wchar_size", i32 4}
!37 = !{i32 8, !"PIC Level", i32 2}
!38 = !{i32 7, !"PIE Level", i32 2}
!39 = !{i32 7, !"uwtable", i32 2}
!40 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!41 = !{!"clang version 18.0.0 (https://github.com/llvm/llvm-project.git 24cf476bd6d144b9fa28325b4d16e3c9dbfc4d4f)"}
!42 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !43, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !46)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !45}
!45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!46 = !{!47}
!47 = !DILocalVariable(name: "cond", arg: 1, scope: !42, file: !1, line: 3, type: !45)
!48 = !DILocation(line: 0, scope: !42)
!49 = !DILocation(line: 4, column: 3, scope: !42)
!50 = !DILocation(line: 6, column: 5, scope: !51)
!51 = distinct !DILexicalBlock(scope: !42, file: !1, line: 4, column: 17)
!52 = !DILocation(line: 8, column: 5, scope: !51)
!53 = !DILocation(line: 10, column: 5, scope: !51)
!54 = !DILocation(line: 12, column: 5, scope: !51)
!55 = !DILocation(line: 14, column: 5, scope: !51)
!56 = !DILocation(line: 16, column: 5, scope: !51)
!57 = !DILocation(line: 18, column: 5, scope: !51)
!58 = !DILocation(line: 20, column: 5, scope: !51)
!59 = !DILocation(line: 22, column: 5, scope: !51)
!60 = !DILocation(line: 24, column: 5, scope: !51)
!61 = !DILocation(line: 26, column: 5, scope: !51)
!62 = !DILocation(line: 28, column: 5, scope: !51)
!63 = !DILocation(line: 30, column: 1, scope: !42)
