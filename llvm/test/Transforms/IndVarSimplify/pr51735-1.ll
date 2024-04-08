; RUN: opt -passes="loop(indvars)" \
; RUN:     --experimental-debuginfo-iterators=false -S -o - < %s | \
; RUN: FileCheck --check-prefix=CHECK %s
; RUN: opt -passes="loop(indvars,loop-deletion)" \
; RUN:     --experimental-debuginfo-iterators=false -S -o - < %s | \
; RUN: FileCheck --check-prefix=CHECK %s

; Make sure that when we delete the loop, that the variable Index has
; the 777 value.

; As this test case does fire the 'indvars' transformation, the debug values
; are added to the 'for.end' exit block. No debug values are preserved by the
; pass to be used by the 'loop-deletion' pass.

; CHECK: for.cond:
; CHECK:   call void @llvm.dbg.value(metadata i32 %[[SSA_INDEX_0:.+]], metadata ![[DBG:[0-9]+]], {{.*}}

; CHECK: for.extra:
; CHECK:   %[[SSA_CALL_0:.+]] = call noundef i32 @"?nop@@YAHH@Z"(i32 noundef %[[SSA_INDEX_0]]), {{.*}}
; CHECK:   br i1 %[[SSA_CMP_0:.+]], label %for.cond, label %if.else, {{.*}}

; CHECK: if.then:
; CHECK:   call void @llvm.dbg.value(metadata i32 777, metadata ![[DBG]], {{.*}}
; CHECK:   call void @llvm.dbg.value(metadata i32 %[[SSA_VAR_1:.+]], metadata ![[VAR:[0-9]+]], {{.*}}
; CHECK:   br label %for.end, {{.*}}

; CHECK: if.else:
; CHECK:   call void @llvm.dbg.value(metadata i32 %[[SSA_VAR_2:.+]], metadata ![[VAR:[0-9]+]], {{.*}}
; CHECK:   br label %for.end, {{.*}}

; CHECK: for.end:
; CHECK:   call void @llvm.dbg.value(metadata i32 777, metadata ![[DBG]], {{.*}}

; CHECK-DAG: ![[DBG]] = !DILocalVariable(name: "Index"{{.*}})
; CHECK-DAG: ![[VAR]] = !DILocalVariable(name: "Var"{{.*}})

define dso_local noundef i32 @"?nop@@YAHH@Z"(i32 noundef %Param) !dbg !11 {
entry:
  %Param.addr = alloca i32, align 4
  store i32 %Param, ptr %Param.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %Param.addr, metadata !32, metadata !DIExpression()), !dbg !35
  ret i32 0, !dbg !36
}

define dso_local void @_Z3barv() local_unnamed_addr #1 !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata i32 777, metadata !16, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 27, metadata !18, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !17
  br label %for.cond, !dbg !20

for.cond:                                         ; preds = %for.cond, %entry
  %Index.0 = phi i32 [ 27, %entry ], [ %inc, %for.extra ], !dbg !17
  call void @llvm.dbg.value(metadata i32 %Index.0, metadata !18, metadata !DIExpression()), !dbg !17
  %cmp = icmp ult i32 %Index.0, 777, !dbg !21
  %inc = add nuw nsw i32 %Index.0, 1, !dbg !24
  call void @llvm.dbg.value(metadata i32 %inc, metadata !18, metadata !DIExpression()), !dbg !17
  br i1 %cmp, label %for.extra, label %if.then, !dbg !25, !llvm.loop !26

for.extra:
  %call.0 = call noundef i32 @"?nop@@YAHH@Z"(i32 noundef %Index.0), !dbg !21
  %cmp.0 = icmp ult i32 %Index.0, %call.0, !dbg !21
  br i1 %cmp.0, label %for.cond, label %if.else, !dbg !25, !llvm.loop !26

if.then:                                          ; preds = %for.cond
  %Var.1 = add nsw i32 %Index.0, 1, !dbg !20
  call void @llvm.dbg.value(metadata i32 %Var.1, metadata !19, metadata !DIExpression()), !dbg !20
  br label %for.end, !dbg !20

if.else:
  %Var.2 = add nsw i32 %Index.0, 2, !dbg !20
  call void @llvm.dbg.value(metadata i32 %Var.2, metadata !19, metadata !DIExpression()), !dbg !20
  br label %for.end, !dbg !20

for.end:                                          ; preds = %if.else, %if.then
  %Zeta.0 = phi i32 [ %Var.1, %if.then ], [ %Var.2, %if.else ], !dbg !20
  call void @llvm.dbg.value(metadata i32 %Zeta.0, metadata !30, metadata !DIExpression()), !dbg !20
  %Var.3 = add nsw i32 %Index.0, 1, !dbg !20
  call void @llvm.dbg.value(metadata i32 %Var.3, metadata !19, metadata !DIExpression()), !dbg !20
  %call = call noundef i32 @"?nop@@YAHH@Z"(i32 noundef %Index.0), !dbg !37
  ret void, !dbg !29
}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0"}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DISubprogram(name: "nop", linkageName: "?nop@@YAHH@Z", scope: !1, file: !1, line: 1, type: !33, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!12 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 5, type: !13, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{}
!16 = !DILocalVariable(name: "End", scope: !12, file: !1, line: 6, type: !10)
!17 = !DILocation(line: 0, scope: !12)
!18 = !DILocalVariable(name: "Index", scope: !12, file: !1, line: 7, type: !10)
!19 = !DILocalVariable(name: "Var", scope: !12, file: !1, line: 8, type: !10)
!20 = !DILocation(line: 9, column: 3, scope: !12)
!21 = !DILocation(line: 9, column: 16, scope: !22)
!22 = distinct !DILexicalBlock(scope: !23, file: !1, line: 9, column: 3)
!23 = distinct !DILexicalBlock(scope: !12, file: !1, line: 9, column: 3)
!24 = !DILocation(line: 9, column: 23, scope: !22)
!25 = !DILocation(line: 9, column: 3, scope: !23)
!26 = distinct !{!26, !25, !27, !28}
!27 = !DILocation(line: 10, column: 5, scope: !23)
!28 = !{!"llvm.loop.mustprogress"}
!29 = !DILocation(line: 12, column: 1, scope: !12)
!30 = !DILocalVariable(name: "Zeta", scope: !12, file: !1, line: 8, type: !10)
!31 = !{!32}
!32 = !DILocalVariable(name: "Param", arg: 1, scope: !11, file: !1, line: 1, type: !10)
!33 = !DISubroutineType(types: !34)
!34 = !{!10, !10}
!35 = !DILocation(line: 1, scope: !11)
!36 = !DILocation(line: 2, scope: !11)
!37 = !DILocation(line: 20, scope: !12)
