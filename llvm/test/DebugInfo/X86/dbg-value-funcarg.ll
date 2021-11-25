; RUN: llc -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=false | FileCheck %s --check-prefixes=COMMON,CHECK
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=true | FileCheck %s --check-prefixes=COMMON,INSTRREF

; Input to this test looked like this and was compiled using: clang -g -O1 -mllvm -stop-after=codegenprepare -S
;
;    extern void bar(int);
;
;    void foo_local(int t1a) {
;      int local = 123;
;      bar(local);
;      local = t1a;
;      bar(local);
;    }
;
;    void foo_other_param(int t2a, int t2b) {
;      bar(t2b);
;      t2b = 123;
;      bar(t2b);
;      t2b = t2a;
;      bar(t2b);
;    }
;
;    void foo_same_param(int t3a) {
;      bar(t3a);
;      int tmp = t3a;
;      t3a = 123;
;      bar(t3a);
;      t3a = tmp;
;      bar(t3a);
;    }
;

; Catch metadata references for involved variables.
;
; COMMON-DAG: ![[T1A:.*]] = !DILocalVariable(name: "t1a"
; COMMON-DAG: ![[LOCAL:.*]] = !DILocalVariable(name: "local"
; COMMON-DAG: ![[T2A:.*]] = !DILocalVariable(name: "t2a"
; COMMON-DAG: ![[T2B:.*]] = !DILocalVariable(name: "t2b"
; COMMON-DAG: ![[T3A:.*]] = !DILocalVariable(name: "t3a"
; COMMON-DAG: ![[TMP:.*]] = !DILocalVariable(name: "tmp"


define dso_local void @foo_local(i32 %t1a) local_unnamed_addr #0 !dbg !7 {
; CHECK-LABEL: name:            foo_local
; CHECK-NOT: DBG_VALUE
; CHECK:      DBG_VALUE $edi, $noreg, ![[T1A]], !DIExpression(),
; CHECK-NEXT: %0:gr32 = COPY $edi
; CHECK-NEXT: DBG_VALUE %0, $noreg, ![[T1A]], !DIExpression(),
; CHECK-NEXT: DBG_VALUE 123, $noreg, ![[LOCAL]], !DIExpression(),
; CHECK-NOT: DBG_VALUE
; CHECK:    CALL64pcrel32 @bar,
; CHECK:    DBG_VALUE %0, $noreg, ![[LOCAL]], !DIExpression(),
; CHECK:    DBG_VALUE $edi, $noreg, ![[T1A]], !DIExpression(),
; CHECK-NOT: DBG_VALUE
; CHECK:    TCRETURNdi64 @bar,
; INSTRREF-LABEL: name:            foo_local
; INSTRREF-NOT: DBG_
; INSTRREF:      DBG_PHI $edi, 1
; INSTRREF:      DBG_VALUE $edi, $noreg, ![[T1A]], !DIExpression(),
; INSTRREF-NEXT: %0:gr32 = COPY $edi
; INSTRREF-NEXT: DBG_VALUE 123, $noreg, ![[LOCAL]], !DIExpression(),
; INSTRREF:      CALL64pcrel32 @bar,
; INSTRREF-NEXT: ADJCALLSTACKUP64
; INSTRREF:      DBG_INSTR_REF 1, 0, ![[LOCAL]], !DIExpression(),
; INSTRREF-NOT: DBG_
; INSTRREF:    TCRETURNdi64 @bar,

entry:
  call void @llvm.dbg.value(metadata i32 %t1a, metadata !12, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 123, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void @bar(i32 123) #3, !dbg !16
  call void @llvm.dbg.value(metadata i32 %t1a, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void @bar(i32 %t1a) #3, !dbg !17
  ret void, !dbg !18
}

define dso_local void @foo_other_param(i32 %t2a, i32 %t2b) local_unnamed_addr #0 !dbg !19 {
; CHECK-LABEL: name:            foo_other_param
; CHECK: DBG_VALUE $edi, $noreg, ![[T2A]], !DIExpression(),
; CHECK: DBG_VALUE $esi, $noreg, ![[T2B]], !DIExpression(),
; CHECK: %1:gr32 = COPY $esi
; CHECK: DBG_VALUE %1, $noreg, ![[T2B]], !DIExpression(),
; CHECK: %0:gr32 = COPY $edi
; CHECK: DBG_VALUE %0, $noreg, ![[T2A]], !DIExpression(),
; CHECK: DBG_VALUE $edi, $noreg, ![[T2B]], !DIExpression(),
; CHECK: CALL64pcrel32 @bar,
; CHECK: DBG_VALUE 123, $noreg, ![[T2B]], !DIExpression(),
; CHECK: CALL64pcrel32 @bar,
; CHECK: DBG_VALUE %0, $noreg, ![[T2B]], !DIExpression(),
; CHECK: DBG_VALUE $edi, $noreg, ![[T2A]], !DIExpression(),
; CHECK: TCRETURNdi64 @bar,
; INSTRREF-LABEL: name:            foo_other_param
; INSTRREF: DBG_PHI $edi, 1
; INSTRREF: DBG_VALUE $edi, $noreg, ![[T2A]], !DIExpression(),
; INSTRREF: DBG_VALUE $esi, $noreg, ![[T2B]], !DIExpression(),
; INSTRREF: CALL64pcrel32 @bar,
; INSTRREF: DBG_VALUE 123, $noreg, ![[T2B]], !DIExpression(),
; INSTRREF: CALL64pcrel32 @bar,
; INSTRREF: DBG_INSTR_REF 1, 0, ![[T2B]], !DIExpression(),
; INSTRREF: TCRETURNdi64 @bar,

entry:
  call void @llvm.dbg.value(metadata i32 %t2a, metadata !23, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 %t2b, metadata !24, metadata !DIExpression()), !dbg !26
  tail call void @bar(i32 %t2b) #3, !dbg !27
  call void @llvm.dbg.value(metadata i32 123, metadata !24, metadata !DIExpression()), !dbg !26
  tail call void @bar(i32 123) #3, !dbg !28
  call void @llvm.dbg.value(metadata i32 %t2a, metadata !24, metadata !DIExpression()), !dbg !26
  tail call void @bar(i32 %t2a) #3, !dbg !29
  ret void, !dbg !30
}

define dso_local void @foo_same_param(i32 %t3a) local_unnamed_addr #0 !dbg !31 {
; CHECK-LABEL: name:            foo_same_param
; CHECK: DBG_VALUE $edi, $noreg, ![[T3A]], !DIExpression(),
; CHECK: %0:gr32 = COPY $edi
; CHECK: DBG_VALUE %0, $noreg, ![[T3A]], !DIExpression(),
; CHECK: CALL64pcrel32 @bar,
; CHECK: DBG_VALUE %0, $noreg, ![[TMP]], !DIExpression(),
; CHECK: DBG_VALUE 123, $noreg, ![[T3A]], !DIExpression(),
; CHECK: CALL64pcrel32 @bar,
; CHECK: DBG_VALUE %0, $noreg, ![[T3A]], !DIExpression(),
; CHECK: TCRETURNdi64 @bar,
; INSTRREF-LABEL: name:            foo_same_param
; INSTRREF: DBG_PHI $edi, 2
; INSTRREF: DBG_PHI $edi, 1
; INSTRREF: DBG_VALUE $edi, $noreg, ![[T3A]], !DIExpression(),
; INSTRREF: CALL64pcrel32 @bar,
; INSTRREF: DBG_INSTR_REF 1, 0, ![[TMP]], !DIExpression(),
; INSTRREF: DBG_VALUE 123, $noreg, ![[T3A]], !DIExpression(),
; INSTRREF: CALL64pcrel32 @bar,
; INSTRREF: DBG_INSTR_REF 2, 0, ![[T3A]], !DIExpression(),
; INSTRREF: TCRETURNdi64 @bar,
entry:
  call void @llvm.dbg.value(metadata i32 %t3a, metadata !33, metadata !DIExpression()), !dbg !35
  tail call void @bar(i32 %t3a) #3, !dbg !36
  call void @llvm.dbg.value(metadata i32 %t3a, metadata !34, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 123, metadata !33, metadata !DIExpression()), !dbg !35
  tail call void @bar(i32 123) #3, !dbg !38
  call void @llvm.dbg.value(metadata i32 %t3a, metadata !33, metadata !DIExpression()), !dbg !35
  tail call void @bar(i32 %t3a) #3, !dbg !39
  ret void, !dbg !40
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

declare dso_local void @bar(i32) local_unnamed_addr

attributes #0 = { nounwind uwtable }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0"}
!7 = distinct !DISubprogram(name: "foo_local", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "t1a", arg: 1, scope: !7, file: !1, line: 3, type: !10)
!13 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 4, type: !10)
!14 = !DILocation(line: 3, column: 20, scope: !7)
!15 = !DILocation(line: 4, column: 7, scope: !7)
!16 = !DILocation(line: 5, column: 3, scope: !7)
!17 = !DILocation(line: 7, column: 3, scope: !7)
!18 = !DILocation(line: 8, column: 1, scope: !7)
!19 = distinct !DISubprogram(name: "foo_other_param", scope: !1, file: !1, line: 10, type: !20, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !22)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !10, !10}
!22 = !{!23, !24}
!23 = !DILocalVariable(name: "t2a", arg: 1, scope: !19, file: !1, line: 10, type: !10)
!24 = !DILocalVariable(name: "t2b", arg: 2, scope: !19, file: !1, line: 10, type: !10)
!25 = !DILocation(line: 10, column: 26, scope: !19)
!26 = !DILocation(line: 10, column: 35, scope: !19)
!27 = !DILocation(line: 11, column: 3, scope: !19)
!28 = !DILocation(line: 13, column: 3, scope: !19)
!29 = !DILocation(line: 15, column: 3, scope: !19)
!30 = !DILocation(line: 16, column: 1, scope: !19)
!31 = distinct !DISubprogram(name: "foo_same_param", scope: !1, file: !1, line: 18, type: !8, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !32)
!32 = !{!33, !34}
!33 = !DILocalVariable(name: "t3a", arg: 1, scope: !31, file: !1, line: 18, type: !10)
!34 = !DILocalVariable(name: "tmp", scope: !31, file: !1, line: 20, type: !10)
!35 = !DILocation(line: 18, column: 25, scope: !31)
!36 = !DILocation(line: 19, column: 3, scope: !31)
!37 = !DILocation(line: 20, column: 7, scope: !31)
!38 = !DILocation(line: 22, column: 3, scope: !31)
!39 = !DILocation(line: 24, column: 3, scope: !31)
!40 = !DILocation(line: 25, column: 1, scope: !31)
