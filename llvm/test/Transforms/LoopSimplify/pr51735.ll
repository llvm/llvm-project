; RUN: opt -passes=indvars -S -o - < %s | FileCheck %s

; Missing local variable 'Index' after loop 'Induction Variable Elimination'.
; When adding a breakpoint at line 11, LLDB does not have information on
; the variable. But it has info on 'Var' and 'End'.

;  1	__attribute__((optnone)) int nop() {
;  2	  return 0;
;  3	}
;  4
;  5	void bar() {
;  6    int End = 777;
;  7	  int Index = 27;
;  8	  char Var = 1;
;  9	  for (; Index < End; ++Index)
; 10	    ;
; 11	  nop();
; 12	}
; 13
; 14	int main () {
; 15	  bar();
; 16	}

; CHECK: for.cond: {{.*}}
; CHECK:   call void @llvm.dbg.value(metadata i32 poison, metadata ![[DBG:[0-9]+]], {{.*}}
; CHECK:   call void @llvm.dbg.value(metadata i32 poison, metadata ![[DBG:[0-9]+]], {{.*}}
; CHECK:   br i1 false, label %for.cond, label %for.end, {{.*}}
; CHECK: for.end: {{.*}}
; CHECK:   call void @llvm.dbg.value(metadata i32 777, metadata ![[DBG:[0-9]+]], {{.*}}
; CHECK:   %call = tail call noundef i32 @_Z3nopv(), {{.*}}
; CHECK-DAG: ![[DBG]] = !DILocalVariable(name: "Index"{{.*}})

define dso_local noundef i32 @_Z3nopv() local_unnamed_addr #0 !dbg !10 {
entry:
  ret i32 0, !dbg !14
}

define dso_local void @_Z3barv() local_unnamed_addr #1 !dbg !15 {
entry:
  call void @llvm.dbg.value(metadata i32 777, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 27, metadata !21, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 1, metadata !22, metadata !DIExpression()), !dbg !20
  br label %for.cond, !dbg !23

for.cond:                                         ; preds = %for.cond, %entry
  %Index.0 = phi i32 [ 27, %entry ], [ %inc, %for.cond ], !dbg !20
  call void @llvm.dbg.value(metadata i32 %Index.0, metadata !21, metadata !DIExpression()), !dbg !20
  %cmp = icmp ult i32 %Index.0, 777, !dbg !24
  %inc = add nuw nsw i32 %Index.0, 1, !dbg !27
  call void @llvm.dbg.value(metadata i32 %inc, metadata !21, metadata !DIExpression()), !dbg !20
  br i1 %cmp, label %for.cond, label %for.end, !dbg !28, !llvm.loop !29

for.end:                                          ; preds = %for.cond
  %call = tail call noundef i32 @_Z3nopv(), !dbg !32
  ret void, !dbg !33
}

define dso_local noundef i32 @main() local_unnamed_addr #2 !dbg !34 {
entry:
  call void @_Z3barv(), !dbg !35
  ret i32 0, !dbg !36
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0 (https://github.com/llvm/llvm-project.git 18c2eb2bf02bd7666523aa566e45d62053b7db80)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0"}
!10 = distinct !DISubprogram(name: "nop", linkageName: "_Z3nopv", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 2, column: 3, scope: !10)
!15 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 5, type: !16, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !{}
!19 = !DILocalVariable(name: "End", scope: !15, file: !1, line: 6, type: !13)
!20 = !DILocation(line: 0, scope: !15)
!21 = !DILocalVariable(name: "Index", scope: !15, file: !1, line: 7, type: !13)
!22 = !DILocalVariable(name: "Var", scope: !15, file: !1, line: 8, type: !13)
!23 = !DILocation(line: 9, column: 3, scope: !15)
!24 = !DILocation(line: 9, column: 16, scope: !25)
!25 = distinct !DILexicalBlock(scope: !26, file: !1, line: 9, column: 3)
!26 = distinct !DILexicalBlock(scope: !15, file: !1, line: 9, column: 3)
!27 = !DILocation(line: 9, column: 23, scope: !25)
!28 = !DILocation(line: 9, column: 3, scope: !26)
!29 = distinct !{!29, !28, !30, !31}
!30 = !DILocation(line: 10, column: 5, scope: !26)
!31 = !{!"llvm.loop.mustprogress"}
!32 = !DILocation(line: 11, column: 3, scope: !15)
!33 = !DILocation(line: 12, column: 1, scope: !15)
!34 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 14, type: !11, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!35 = !DILocation(line: 15, column: 3, scope: !34)
!36 = !DILocation(line: 16, column: 1, scope: !34)
