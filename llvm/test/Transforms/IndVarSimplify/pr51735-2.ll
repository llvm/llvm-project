; RUN: opt -passes="loop(indvars)" \
; RUN:     --experimental-debuginfo-iterators=false -S -o - < %s | \
; RUN: FileCheck --implicit-check-not="call void @llvm.dbg" \
; RUN:           --check-prefix=ALL-CHECK --check-prefix=PRE-CHECK %s
; RUN: opt -passes="loop(indvars,loop-deletion)" \
; RUN:     --experimental-debuginfo-iterators=false -S -o - < %s | \
; RUN: FileCheck --implicit-check-not="call void @llvm.dbg" \
; RUN:           --check-prefix=ALL-CHECK --check-prefix=POST-CHECK %s

; Check what happens to a modified but otherwise unused variable in a loop
; that gets deleted. The assignment in the loop is 'forgotten' by LLVM and
; doesn't appear in the debugging information. This behaviour is suboptimal,
; but we want to know if it changes

; For all cases, LLDB shows
;   Var = <no location, value may have been optimized out>

;  1	__attribute__((optnone)) int nop() {
;  2	  return 0;
;  3	}
;  4
;  5	void bar() {
;  6    int End = 777;
;  7	  int Index = 27;
;  8	  char Var = 1;
;  9	  for (; Index < End; ++Index) {
; 10      if (Index == 666) {
; 11        ++Var;
; 12      }
; 13    }
; 14	  nop();
; 15	}

; ALL-CHECK: entry:
; ALL-CHECK:   call void @llvm.dbg.value(metadata i32 1, metadata ![[DBG:[0-9]+]], {{.*}}

; Only the 'indvars' pass is executed.
; PRE-CHECK: for.cond:
; PRE-CHECK:   call void @llvm.dbg.value(metadata i32 %[[SSA_VAR_0:.+]], metadata ![[DBG]], {{.*}}
; PRE-CHECK:   call void @llvm.dbg.value(metadata !DIArgList{{.*}}

; PRE-CHECK: for.body:
; PRE-CHECK:   {{.*}} = icmp eq i32 %[[SSA_INDEX_0:.+]], 666
; PRE-CHECK:   {{.*}} = add nsw i32 %[[SSA_VAR_0]], 1
; PRE-CHECK:   {{.*}} = select i1 {{.*}}, i32 {{.*}}, i32 %[[SSA_VAR_0]]
; PRE-CHECK:   call void @llvm.dbg.value(metadata i32 {{.*}}, metadata ![[DBG]], {{.*}}
; PRE-CHECK:   br label %for.cond

; PRE-CHECK: for.end:
; PRE-CHECK:   ret void
; PRE-CHECK-DAG: ![[DBG]] = !DILocalVariable(name: "Var"{{.*}})

; The 'indvars' and 'loop-deletion' passes are executed.
; POST-CHECK: for.end:
; POST-CHECK:   call void @llvm.dbg.value(metadata i32 undef, metadata ![[DBG:[0-9]+]], {{.*}}
; POST-CHECK:   ret void
; POST-CHECK-DAG: ![[DBG]] = !DILocalVariable(name: "Var"{{.*}})

define dso_local void @_Z3barv() local_unnamed_addr !dbg !18 {
entry:
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !22
  br label %for.cond, !dbg !25

for.cond:                                         ; preds = %for.cond, %entry
  %Index.0 = phi i32 [ 27, %entry ], [ %inc2, %for.body ], !dbg !22
  %Var.0 = phi i32 [ 1, %entry ], [ %spec.select, %for.body ], !dbg !22
  call void @llvm.dbg.value(metadata i32 %Var.0, metadata !24, metadata !DIExpression()), !dbg !22
  %cmp = icmp ult i32 %Index.0, 777, !dbg !26
  call void @llvm.dbg.value(metadata !DIArgList(i32 poison, i32 %Index.0), metadata !24, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 666, DW_OP_eq, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_plus, DW_OP_stack_value)), !dbg !22
  %inc2 = add nuw nsw i32 %Index.0, 1, !dbg !29
  br i1 %cmp, label %for.body, label %for.end, !dbg !30, !llvm.loop !31

for.body:                                         ; preds = %for.cond
  %cmp1 = icmp eq i32 %Index.0, 666, !dbg !30
  %inc = add nsw i32 %Var.0, 1
  %spec.select = select i1 %cmp1, i32 %inc, i32 %Var.0, !dbg !32
  call void @llvm.dbg.value(metadata i32 %spec.select, metadata !24, metadata !DIExpression()), !dbg !22
  br label %for.cond, !dbg !34, !llvm.loop !35

for.end:                                          ; preds = %for.cond
  ret void, !dbg !35
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test-b.cpp", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 19.0.0"}
!10 = distinct !DISubprogram(name: "nop", linkageName: "_Z3nopi", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "Param", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!16 = !DILocation(line: 1, column: 38, scope: !10)
!17 = !DILocation(line: 2, column: 3, scope: !10)
!18 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 5, type: !19, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!22 = !DILocation(line: 0, scope: !18)
!24 = !DILocalVariable(name: "Var", scope: !18, file: !1, line: 8, type: !13)
!25 = !DILocation(line: 9, column: 3, scope: !18)
!26 = !DILocation(line: 9, column: 16, scope: !27)
!27 = distinct !DILexicalBlock(scope: !28, file: !1, line: 9, column: 3)
!28 = distinct !DILexicalBlock(scope: !18, file: !1, line: 9, column: 3)
!29 = !DILocation(line: 9, column: 23, scope: !27)
!30 = !DILocation(line: 9, column: 3, scope: !28)
!31 = distinct !{!31, !30, !32, !33}
!32 = !DILocation(line: 11, column: 9, scope: !28)
!33 = !{!"llvm.loop.mustprogress"}
!34 = !DILocation(line: 12, column: 3, scope: !18)
!35 = !DILocation(line: 13, column: 1, scope: !18)
!36 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 15, type: !37, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!37 = !DISubroutineType(types: !38)
!38 = !{!13}
!39 = !DILocation(line: 16, column: 3, scope: !36)
!40 = !DILocation(line: 17, column: 1, scope: !36)
