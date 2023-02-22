; RUN: opt %s -S -passes=sroa -o - | FileCheck %s

;; Check that SROA preserves the InlinedAt status of new dbg.assign intriniscs
;; it inserts.

;; $cat test.c
;; typedef struct {
;;   int a;
;;   int b[];
;; } c;
;; int d, e, f;
;; void g(c *h) {
;;   if (d)
;;     h->a = 1;
;; }
;; void i(c *h) {
;;   long j = f = 0;
;;   for (; f < h->a; f++)
;;     j += h->b[f];
;;   e = j;
;; }
;; void k() {
;;   c j;
;;   g(&j);
;;   i(&j);
;; }
;; void l() { k(); }
;;
;; $ clang test.c -Xclang -fexperimental-assignment-tracking  -O2 -g

; CHECK: call void @llvm.dbg.assign(metadata i1 false, metadata !{{.+}}, metadata !DIExpression(), metadata !{{.+}}, metadata ptr undef, metadata !DIExpression()), !dbg ![[DBG:[0-9]+]]

; CHECK-DAG: ![[DBG]] = !DILocation(line: 0, scope: ![[INL_SC:[0-9]+]], inlinedAt: ![[IA:[0-9]+]])
; CHECK-DAG: ![[IA]] = distinct !DILocation(line: 21, column: 12, scope: ![[SC:[0-9]+]])
; CHECK-DAG: ![[SC]] = distinct !DISubprogram(name: "l",
; CHECK-DAG: ![[INL_SC]] = distinct !DISubprogram(name: "k"

%struct.c = type { i32, [0 x i32] }

@f = dso_local local_unnamed_addr global i32 0, align 4, !dbg !9
@e = dso_local local_unnamed_addr global i32 0, align 4, !dbg !6

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #1
declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #2
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture) #2

define dso_local void @l() local_unnamed_addr #4 !dbg !73 {
entry:
  %j.i = alloca %struct.c, align 4, !DIAssignID !74
  ; NOTE: this has been changed from undef to false so that the intrinsic isn't
  ; deleted as redundant.
  call void @llvm.dbg.assign(metadata i1 false, metadata !64, metadata !DIExpression(), metadata !74, metadata ptr %j.i, metadata !DIExpression()) #5, !dbg !75
  %0 = bitcast ptr %j.i to ptr, !dbg !77
  call void @llvm.lifetime.start.p0i8(i64 4, ptr nonnull %0) #5, !dbg !77
  %arrayidx.i.i = getelementptr inbounds %struct.c, ptr %j.i, i64 0, i32 1, i64 0, !dbg !78
  %1 = load i32, ptr %arrayidx.i.i, align 4, !dbg !78
  store i32 1, ptr @f, align 4, !dbg !80
  store i32 %1, ptr @e, align 4, !dbg !81
  call void @llvm.lifetime.end.p0i8(i64 4, ptr nonnull %0) #5, !dbg !82
  ret void, !dbg !83
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !1000}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "d", scope: !2, file: !3, line: 5, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0, !6, !9}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "e", scope: !2, file: !3, line: 5, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "f", scope: !2, file: !3, line: 5, type: !8, isLocal: false, isDefinition: true)
!11 = !{i32 7, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 12.0.0)"}
!15 = distinct !DISubprogram(name: "g", scope: !3, file: !3, line: 6, type: !16, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !27)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "c", file: !3, line: 4, baseType: !20)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 32, elements: !21)
!21 = !{!22, !23}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !20, file: !3, line: 2, baseType: !8, size: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !20, file: !3, line: 3, baseType: !24, offset: 32)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, elements: !25)
!25 = !{!26}
!26 = !DISubrange(count: -1)
!27 = !{!28}
!28 = !DILocalVariable(name: "h", arg: 1, scope: !15, file: !3, line: 6, type: !18)
!29 = !DILocation(line: 7, column: 7, scope: !30)
!30 = distinct !DILexicalBlock(scope: !15, file: !3, line: 7, column: 7)
!35 = !DILocation(line: 7, column: 7, scope: !15)
!36 = !DILocation(line: 8, column: 8, scope: !30)
!37 = !DILocation(line: 8, column: 10, scope: !30)
!38 = !DILocation(line: 8, column: 5, scope: !30)
!39 = !DILocation(line: 9, column: 1, scope: !15)
!40 = distinct !DISubprogram(name: "i", scope: !3, file: !3, line: 10, type: !16, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !41)
!41 = !{!42, !43}
!42 = !DILocalVariable(name: "h", arg: 1, scope: !40, file: !3, line: 10, type: !18)
!43 = !DILocalVariable(name: "j", scope: !40, file: !3, line: 11, type: !44)
!44 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!45 = !DILocation(line: 0, scope: !40)
!46 = !DILocation(line: 12, column: 17, scope: !47)
!47 = distinct !DILexicalBlock(scope: !48, file: !3, line: 12, column: 3)
!48 = distinct !DILexicalBlock(scope: !40, file: !3, line: 12, column: 3)
!49 = !DILocation(line: 12, column: 12, scope: !47)
!50 = !DILocation(line: 12, column: 3, scope: !48)
!51 = !DILocation(line: 13, column: 10, scope: !47)
!52 = !DILocation(line: 13, column: 7, scope: !47)
!53 = !DILocation(line: 12, column: 21, scope: !47)
!54 = distinct !{!54, !50, !55, !56}
!55 = !DILocation(line: 13, column: 16, scope: !48)
!56 = !{!"llvm.loop.mustprogress"}
!57 = !DILocation(line: 14, column: 7, scope: !40)
!58 = !DILocation(line: 14, column: 5, scope: !40)
!59 = !DILocation(line: 15, column: 1, scope: !40)
!60 = distinct !DISubprogram(name: "k", scope: !3, file: !3, line: 16, type: !61, scopeLine: 16, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !63)
!61 = !DISubroutineType(types: !62)
!62 = !{null}
!63 = !{!64}
!64 = !DILocalVariable(name: "j", scope: !60, file: !3, line: 17, type: !19)
!65 = distinct !DIAssignID()
!66 = !DILocation(line: 0, scope: !60)
!67 = !DILocation(line: 17, column: 3, scope: !60)
!68 = !DILocation(line: 13, column: 10, scope: !47, inlinedAt: !69)
!69 = distinct !DILocation(line: 19, column: 3, scope: !60)
!70 = !DILocation(line: 0, scope: !40, inlinedAt: !69)
!71 = !DILocation(line: 14, column: 5, scope: !40, inlinedAt: !69)
!72 = !DILocation(line: 20, column: 1, scope: !60)
!73 = distinct !DISubprogram(name: "l", scope: !3, file: !3, line: 21, type: !61, scopeLine: 21, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!74 = distinct !DIAssignID()
!75 = !DILocation(line: 0, scope: !60, inlinedAt: !76)
!76 = distinct !DILocation(line: 21, column: 12, scope: !73)
!77 = !DILocation(line: 17, column: 3, scope: !60, inlinedAt: !76)
!78 = !DILocation(line: 13, column: 10, scope: !47, inlinedAt: !79)
!79 = distinct !DILocation(line: 19, column: 3, scope: !60, inlinedAt: !76)
!80 = !DILocation(line: 0, scope: !40, inlinedAt: !79)
!81 = !DILocation(line: 14, column: 5, scope: !40, inlinedAt: !79)
!82 = !DILocation(line: 20, column: 1, scope: !60, inlinedAt: !76)
!83 = !DILocation(line: 21, column: 17, scope: !73)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
