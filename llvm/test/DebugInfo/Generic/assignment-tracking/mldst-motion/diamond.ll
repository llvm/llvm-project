; RUN: opt -passes=mldst-motion -S %s -o - -experimental-assignment-tracking \
; RUN: | FileCheck %s

;; $ cat test.cpp
;; int cond;
;; void esc(int*);
;; void fun() {
;;   int a[4];
;;   if (cond)
;;     a[1] = 1;
;;   else
;;     a[1] = 2;
;;   esc(a);
;; }
;;
;;
;; mldst-motion will merge and sink the stores in if.then and if.else into
;; if.end. Ensure the merged store has a DIAssignID linking it to the two
;; dbg.assign intrinsics which have been left in their original blocks.

; CHECK: if.then:
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 1,{{.+}}, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata ![[ID:[0-9]+]], metadata ptr %1, metadata !DIExpression())

; CHECK: if.else:
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 2,{{.+}}, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata ![[ID]], metadata ptr %1, metadata !DIExpression())

; CHECK: if.end:
; CHECK: store i32 %.sink, ptr %1, align 4{{.+}}, !DIAssignID ![[ID]]

@cond = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

define dso_local void @_Z3funv() !dbg !11 {
entry:
  %a = alloca [4 x i32], align 16, !DIAssignID !19
  call void @llvm.dbg.assign(metadata i1 undef, metadata !15, metadata !DIExpression(), metadata !19, metadata ptr %a, metadata !DIExpression()), !dbg !20
  call void @llvm.lifetime.start.p0i8(i64 16, ptr nonnull %a) #4, !dbg !21
  %0 = load i32, i32* @cond, align 4, !dbg !22
  %tobool.not = icmp eq i32 %0, 0, !dbg !22
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !28

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 1, !dbg !29
  store i32 1, ptr %arrayidx, align 4, !dbg !30, !DIAssignID !31
  call void @llvm.dbg.assign(metadata i32 1, metadata !15, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !31, metadata ptr %arrayidx, metadata !DIExpression()), !dbg !20
  br label %if.end, !dbg !29

if.else:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 1, !dbg !32
  store i32 2, ptr %arrayidx1, align 4, !dbg !33, !DIAssignID !34
  call void @llvm.dbg.assign(metadata i32 2, metadata !15, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !34, metadata ptr %arrayidx1, metadata !DIExpression()), !dbg !20
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %arraydecay = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 0, !dbg !35
  call void @_Z3escPi(ptr noundef nonnull %arraydecay), !dbg !36
  call void @llvm.lifetime.end.p0i8(i64 16, ptr nonnull %a) #4, !dbg !37
  ret void, !dbg !37
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare !dbg !38 dso_local void @_Z3escPi(i32* noundef)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "cond", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{!"clang version 14.0.0"}
!11 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{!15}
!15 = !DILocalVariable(name: "a", scope: !11, file: !3, line: 4, type: !16)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 128, elements: !17)
!17 = !{!18}
!18 = !DISubrange(count: 4)
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 0, scope: !11)
!21 = !DILocation(line: 4, column: 3, scope: !11)
!22 = !DILocation(line: 5, column: 7, scope: !23)
!23 = distinct !DILexicalBlock(scope: !11, file: !3, line: 5, column: 7)
!28 = !DILocation(line: 5, column: 7, scope: !11)
!29 = !DILocation(line: 6, column: 5, scope: !23)
!30 = !DILocation(line: 6, column: 10, scope: !23)
!31 = distinct !DIAssignID()
!32 = !DILocation(line: 8, column: 5, scope: !23)
!33 = !DILocation(line: 8, column: 10, scope: !23)
!34 = distinct !DIAssignID()
!35 = !DILocation(line: 9, column: 7, scope: !11)
!36 = !DILocation(line: 9, column: 3, scope: !11)
!37 = !DILocation(line: 10, column: 1, scope: !11)
!38 = !DISubprogram(name: "esc", linkageName: "_Z3escPi", scope: !3, file: !3, line: 2, type: !39, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !42)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !41}
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!42 = !{}
