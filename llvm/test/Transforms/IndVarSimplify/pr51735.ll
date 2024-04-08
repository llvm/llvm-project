; RUN: opt -passes="loop(indvars)" \
; RUN:     --experimental-debuginfo-iterators=false -S -o - < %s | \
; RUN: FileCheck --check-prefix=PRE-CHECK %s
; RUN: opt -passes="loop(indvars,loop-deletion)" \
; RUN:     --experimental-debuginfo-iterators=false -S -o - < %s | \
; RUN: FileCheck --check-prefix=POST-CHECK %s

; Make sure that when we delete the loop in the code below, that the variable
; Index has the 777 value.

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

; Only the 'indvars' pass is executed.
; As this test case does not fire the 'indvars' transformation, no debug values
; are preserved and or added.

; PRE-CHECK: for.cond:
; PRE-CHECK:   call void @llvm.dbg.value(metadata i32 poison, metadata ![[DBG_1:[0-9]+]], {{.*}}
; PRE-CHECK:   call void @llvm.dbg.value(metadata i32 poison, metadata ![[DBG_1]], {{.*}}
; PRE-CHECK:   br i1 false, label %for.cond, label %for.end

; PRE-CHECK: for.end:
; PRE-CHECK-NOT: call void @llvm.dbg.value
; PRE-CHECK:   ret void
; PRE-CHECK-DAG: ![[DBG_1]] = !DILocalVariable(name: "Index"{{.*}})

; The 'indvars' and 'loop-deletion' passes are executed.
; The loop is deleted and the debug values collected by 'indvars' are used by
; 'loop-deletion' to add the induction variable debug value.

; POST-CHECK: for.end:
; POST-CHECK:   call void @llvm.dbg.value(metadata i32 777, metadata ![[DBG_2:[0-9]+]], {{.*}}
; POST-CHECK:   ret void
; POST-CHECK-DAG: ![[DBG_2]] = !DILocalVariable(name: "Index"{{.*}})

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
  ret void, !dbg !33
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
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
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
!33 = !DILocation(line: 12, column: 1, scope: !15)
