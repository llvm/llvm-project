; RUN: llc %s -stop-after=finalize-isel -o - -experimental-assignment-tracking \
; RUN: | FileCheck %s --implicit-check-not=DBG

;; Tiny loop with a store sunk out of it:
;; void e();
;; void es(int*);
;; int *g;
;; int getInt();
;; int f(int a, int b) {
;;   int z = getInt();
;;   while (g) {
;;     e();
;;     a = z;
;;   }
;;   es(&a);
;;   return a;
;; }
;;
;; Store to `a` has been sunk out the loop - there's a dbg.assign left in the
;; loop that is linked the store that is now outside it. Check that the memory
;; location is not used inside the loop and is reinstated after the sunk store.

; CHECK-DAG: ![[A:[0-9]+]] = !DILocalVariable(name: "a",

; CHECK-LABEL: bb.0.entry:
; CHECK: DBG_VALUE $edi, $noreg, ![[A]], !DIExpression()
; CHECK: CALL64pcrel32 @getInt{{.*}}debug-instr-number 1

; CHECK-LABEL: bb.2.while.body:
; CHECK: DBG_INSTR_REF 1, 6, ![[A]], !DIExpression()

; CHECK-LABEL: bb.3.while.end:
; CHECK: MOV32mr %stack.0.a.addr, 1, $noreg, 0, $noreg, %1
; CHECK-NEXT: DBG_VALUE %stack.0.a.addr, $noreg, ![[A]], !DIExpression(DW_OP_deref)

target triple = "x86_64-unknown-linux-gnu"

@g = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z1fii(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 !dbg !12 {
entry:
  %a.addr = alloca i32, align 4, !DIAssignID !18
  call void @llvm.dbg.assign(metadata i1 undef, metadata !16, metadata !DIExpression(), metadata !18, metadata ptr %a.addr, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.assign(metadata i32 %a, metadata !16, metadata !DIExpression(), metadata !20, metadata ptr %a.addr, metadata !DIExpression()), !dbg !19
  %z = call i32 @getInt()
  %0 = load ptr, ptr @g, align 8, !dbg !22
  %tobool.not1 = icmp eq ptr %0, null, !dbg !22
  br i1 %tobool.not1, label %while.end, label %while.body, !dbg !27

while.body:                                       ; preds = %entry, %while.body
  tail call void @_Z1ev(), !dbg !28
  call void @llvm.dbg.assign(metadata i32 %z, metadata !16, metadata !DIExpression(), metadata !20, metadata ptr %a.addr, metadata !DIExpression()), !dbg !19
  %1 = load ptr, ptr @g, align 8, !dbg !22
  %tobool.not = icmp eq ptr %1, null, !dbg !22
  br i1 %tobool.not, label %while.end, label %while.body, !dbg !27, !llvm.loop !30

while.end:                                        ; preds = %while.body, %entry
  %storemerge.lcssa = phi i32 [ %a, %entry ], [ %b, %while.body ]
  store i32 %storemerge.lcssa, ptr %a.addr, align 4, !DIAssignID !20
  call void @_Z2esPi(ptr noundef nonnull %a.addr), !dbg !35
  %2 = load i32, ptr %a.addr, align 4, !dbg !36
  %r = add i32 %2, %z
  ret i32 %r, !dbg !37
}

declare !dbg !38 dso_local void @_Z1ev() local_unnamed_addr #1
declare !dbg !42 dso_local void @_Z2esPi(ptr noundef) local_unnamed_addr #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2
declare dso_local i32 @getInt()

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
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
!12 = distinct !DISubprogram(name: "f", linkageName: "_Z1fii", scope: !3, file: !3, line: 5, type: !13, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!6, !6, !6}
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !3, line: 5, type: !6)
!17 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !3, line: 5, type: !6)
!18 = distinct !DIAssignID()
!19 = !DILocation(line: 0, scope: !12)
!20 = distinct !DIAssignID()
!21 = distinct !DIAssignID()
!22 = !DILocation(line: 6, column: 10, scope: !12)
!27 = !DILocation(line: 6, column: 3, scope: !12)
!28 = !DILocation(line: 7, column: 5, scope: !29)
!29 = distinct !DILexicalBlock(scope: !12, file: !3, line: 6, column: 13)
!30 = distinct !{!30, !27, !31, !32}
!31 = !DILocation(line: 9, column: 3, scope: !12)
!32 = !{!"llvm.loop.mustprogress"}
!35 = !DILocation(line: 10, column: 3, scope: !12)
!36 = !DILocation(line: 11, column: 10, scope: !12)
!37 = !DILocation(line: 11, column: 3, scope: !12)
!38 = !DISubprogram(name: "e", linkageName: "_Z1ev", scope: !3, file: !3, line: 2, type: !39, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !41)
!39 = !DISubroutineType(types: !40)
!40 = !{null}
!41 = !{}
!42 = !DISubprogram(name: "es", linkageName: "_Z2esPi", scope: !3, file: !3, line: 3, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !41)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !5}
