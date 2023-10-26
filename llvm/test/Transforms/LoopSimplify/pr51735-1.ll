; RUN: opt -passes=indvars -S -o - < %s | FileCheck %s

; Missing local variable 'Index' after loop 'Induction Variable Elimination'.
; When adding a breakpoint at line 11, LLDB does not have information on
; the variable. But it has info on 'Var' and 'End'.

;  1	__attribute__((optnone)) int nop(int Param) {
;  2	  return 0;
;  3	}
;  4
;  5	void bar() {
;  6    int End = 777;
;  7	  int Index = 27;
;  8	  char Var = 1;
;  9	  for (; Index < End; ++Index)
; 10	    ;
; 11	  nop(Index);
; 12	}
; 13
; 14	int main () {
; 15	  bar();
; 16	}

; CHECK: for.cond: {{.*}}
; CHECK:   call void @llvm.dbg.value(metadata i32 %Index.{{[0-9]+}}, metadata ![[DBG:[0-9]+]], {{.*}}
; CHECK:   call void @llvm.dbg.value(metadata i32 %inc, metadata ![[DBG:[0-9]+]], {{.*}}
; CHECK: for.end: {{.*}}
; CHECK:   call void @llvm.dbg.value(metadata i32 777, metadata ![[DBG:[0-9]+]], {{.*}}
; CHECK-DAG: ![[DBG]] = !DILocalVariable(name: "Index"{{.*}})

define dso_local void @_Z3barv() local_unnamed_addr #2 !dbg !18 {
entry:
  call void @llvm.dbg.value(metadata i32 777, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 27, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !22
  br label %for.cond, !dbg !25

for.cond:                                         ; preds = %for.cond, %entry
  %Index.0 = phi i32 [ 27, %entry ], [ %inc, %for.cond ], !dbg !22
  call void @llvm.dbg.value(metadata i32 %Index.0, metadata !23, metadata !DIExpression()), !dbg !22
  %cmp = icmp ult i32 %Index.0, 777, !dbg !26
  %inc = add nuw nsw i32 %Index.0, 1, !dbg !29
  call void @llvm.dbg.value(metadata i32 %inc, metadata !23, metadata !DIExpression()), !dbg !22
  br i1 %cmp, label %for.cond, label %for.end, !dbg !30, !llvm.loop !31

for.end:                                          ; preds = %for.cond
  %Index.0.lcssa = phi i32 [ %Index.0, %for.cond ], !dbg !22
  ret void, !dbg !35
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

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
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!18 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 5, type: !19, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !DILocalVariable(name: "End", scope: !18, file: !1, line: 6, type: !13)
!22 = !DILocation(line: 0, scope: !18)
!23 = !DILocalVariable(name: "Index", scope: !18, file: !1, line: 7, type: !13)
!24 = !DILocalVariable(name: "Var", scope: !18, file: !1, line: 8, type: !13)
!25 = !DILocation(line: 9, column: 3, scope: !18)
!26 = !DILocation(line: 9, column: 16, scope: !27)
!27 = distinct !DILexicalBlock(scope: !28, file: !1, line: 9, column: 3)
!28 = distinct !DILexicalBlock(scope: !18, file: !1, line: 9, column: 3)
!29 = !DILocation(line: 9, column: 23, scope: !27)
!30 = !DILocation(line: 9, column: 3, scope: !28)
!31 = distinct !{!31, !30, !32, !33}
!32 = !DILocation(line: 10, column: 5, scope: !28)
!33 = !{!"llvm.loop.mustprogress"}
!34 = !DILocation(line: 11, column: 3, scope: !18)
!35 = !DILocation(line: 12, column: 1, scope: !18)
