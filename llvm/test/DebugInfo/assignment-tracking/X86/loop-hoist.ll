; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_


; RUN: llc --try-experimental-debuginfo-iterators %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; $ cat test.cpp
;; int d();
;; void e();
;; void es(int*);
;; int *g;
;; int f(int a, int b, int c) {
;;   do {
;;     /* stuff */
;;     c *= c;
;;     a = b;
;;     e();
;;   } while (d());
;;   es(&a);
;;   return a + c;
;; }

;; The variable of interest is `a`, which has a store that is hoisted out of the
;; loop into the entry BB. Check the memory location is not used after the
;; hoisted store until the assignment position within the loop.

; CHECK-DAG: ![[A:[0-9]+]] = !DILocalVariable(name: "a",

; CHECK: bb.0.entry:
; CHECK:      DBG_VALUE $edi, $noreg, ![[A]], !DIExpression()

; CHECK: bb.1.do.body:
; CHECK: DBG_VALUE %stack.0.a.addr, $noreg, ![[A]], !DIExpression(DW_OP_deref)

target triple = "x86_64-unknown-linux-gnu"

@g = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0

define dso_local noundef i32 @_Z1fiii(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 !dbg !12 {
entry:
  %a.addr = alloca i32, align 4, !DIAssignID !19
  call void @llvm.dbg.assign(metadata i1 undef, metadata !16, metadata !DIExpression(), metadata !19, metadata ptr %a.addr, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.assign(metadata i32 %a, metadata !16, metadata !DIExpression(), metadata !21, metadata ptr %a.addr, metadata !DIExpression()), !dbg !20
  store i32 %b, ptr %a.addr, align 4, !DIAssignID !28
  br label %do.body, !dbg !29

do.body:                                          ; preds = %do.body, %entry
  %c.addr.0 = phi i32 [ %c, %entry ], [ %mul, %do.body ]
  %mul = mul nsw i32 %c.addr.0, %c.addr.0, !dbg !30
  call void @llvm.dbg.assign(metadata i32 %b, metadata !16, metadata !DIExpression(), metadata !28, metadata ptr %a.addr, metadata !DIExpression()), !dbg !20
  tail call void @_Z1ev(), !dbg !33
  %call = tail call noundef i32 @_Z1dv(), !dbg !34
  %tobool.not = icmp eq i32 %call, 0, !dbg !34
  br i1 %tobool.not, label %do.end, label %do.body, !dbg !35, !llvm.loop !36

do.end:                                           ; preds = %do.body
  call void @_Z2esPi(ptr noundef nonnull %a.addr), !dbg !39
  %0 = load i32, ptr %a.addr, align 4, !dbg !40
  %add = add nsw i32 %0, %mul, !dbg !41
  ret i32 %add, !dbg !42
}

declare !dbg !43 dso_local void @_Z1ev() local_unnamed_addr #1
declare !dbg !47 dso_local noundef i32 @_Z1dv() local_unnamed_addr #1
declare !dbg !50 dso_local void @_Z2esPi(ptr noundef) local_unnamed_addr #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !1000}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"uwtable", i32 1}
!11 = !{!"clang version 14.0.0"}
!12 = distinct !DISubprogram(name: "f", linkageName: "_Z1fiii", scope: !3, file: !3, line: 5, type: !13, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!6, !6, !6, !6}
!15 = !{!16, !17, !18}
!16 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !3, line: 5, type: !6)
!17 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !3, line: 5, type: !6)
!18 = !DILocalVariable(name: "c", arg: 3, scope: !12, file: !3, line: 5, type: !6)
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 0, scope: !12)
!21 = distinct !DIAssignID()
!22 = distinct !DIAssignID()
!23 = distinct !DIAssignID()
!28 = distinct !DIAssignID()
!29 = !DILocation(line: 6, column: 3, scope: !12)
!30 = !DILocation(line: 8, column: 7, scope: !31)
!31 = distinct !DILexicalBlock(scope: !12, file: !3, line: 6, column: 6)
!32 = distinct !DIAssignID()
!33 = !DILocation(line: 10, column: 5, scope: !31)
!34 = !DILocation(line: 11, column: 12, scope: !12)
!35 = !DILocation(line: 11, column: 3, scope: !31)
!36 = distinct !{!36, !29, !37, !38}
!37 = !DILocation(line: 11, column: 15, scope: !12)
!38 = !{!"llvm.loop.mustprogress"}
!39 = !DILocation(line: 12, column: 3, scope: !12)
!40 = !DILocation(line: 13, column: 10, scope: !12)
!41 = !DILocation(line: 13, column: 12, scope: !12)
!42 = !DILocation(line: 13, column: 3, scope: !12)
!43 = !DISubprogram(name: "e", linkageName: "_Z1ev", scope: !3, file: !3, line: 2, type: !44, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !46)
!44 = !DISubroutineType(types: !45)
!45 = !{null}
!46 = !{}
!47 = !DISubprogram(name: "d", linkageName: "_Z1dv", scope: !3, file: !3, line: 1, type: !48, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !46)
!48 = !DISubroutineType(types: !49)
!49 = !{!6}
!50 = !DISubprogram(name: "es", linkageName: "_Z2esPi", scope: !3, file: !3, line: 3, type: !51, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !46)
!51 = !DISubroutineType(types: !52)
!52 = !{null, !5}
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
