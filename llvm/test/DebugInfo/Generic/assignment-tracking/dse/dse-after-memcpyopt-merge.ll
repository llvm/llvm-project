; RUN: opt %s -S -passes=dse -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators %s -S -passes=dse -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Observed in the wild, but test is created by running memcpyopt on
;; assignment-tracking/memcpyopt/merge-stores.ll then manually inserting
;; two stores that overwrite each end of the memset.
;;
;; A memory intrinsic (or vector instruction) might have multiple dbg.assigns
;; linked to it if it has been created as a result of merging scalar stores,
;; such as in this example. DSE is going to shorten the memset because there's
;; a later store that overwrites part of it. Unlink the dbg.assigns that
;; describe the overlapping fragments.

;; Check that there's an unlinked dbg.assign inserted after each overlapping
;; fragment of the shortened store.
;;
; CHECK: #dbg_assign({{.*}}, ptr %g, !DIExpression(),
; CHECK: #dbg_assign(float 0.000000e+00, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 64, 32), ![[ID:[0-9]+]], ptr %arrayidx.i, !DIExpression(),
; CHECK: #dbg_assign(float 0.000000e+00, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 32, 32), ![[ID]], ptr %arrayidx3.i, !DIExpression(),
; CHECK: #dbg_assign(float 0.000000e+00, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 32), ![[UniqueID1:[0-9]+]], ptr undef, !DIExpression(),
; CHECK: #dbg_assign(float 0.000000e+00, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 96, 32), ![[UniqueID2:[0-9]+]], ptr undef, !DIExpression(),
; CHECK: call void @llvm.memset{{.*}}, !DIAssignID ![[ID]]

; CHECK-DAG: ![[ID]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID1]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID2]] = distinct !DIAssignID()

%struct.v = type { [4 x float] }

$_ZN1vC2Ef = comdat any

define dso_local void @_Z1fv() local_unnamed_addr !dbg !7 {
entry:
  %g = alloca %struct.v, align 4, !DIAssignID !23
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !23, metadata ptr %g, metadata !DIExpression()), !dbg !24
   %arrayidx.i = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 2, !dbg !37
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !39, metadata ptr %arrayidx.i, metadata !DIExpression()), !dbg !24
  %arrayidx3.i = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 1, !dbg !40
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !39, metadata ptr %arrayidx3.i, metadata !DIExpression()), !dbg !24
  %arrayidx5.i = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 0, !dbg !41
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !39, metadata ptr %arrayidx5.i, metadata !DIExpression()), !dbg !24
  %arrayidx7.i = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 3, !dbg !42
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 96, 32), metadata !39, metadata ptr %arrayidx7.i, metadata !DIExpression()), !dbg !24
  %0 = bitcast ptr %arrayidx5.i to ptr, !dbg !43
  call void @llvm.memset.p0.i64(ptr align 4 %0, i8 0, i64 16, i1 false), !dbg !44, !DIAssignID !39
  ;; -- Start modification
  %arrayidx7 = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 3, !dbg !24
  store float 0.000000e+00, ptr %arrayidx7, align 4, !dbg !24, !DIAssignID !49
  %arrayidx = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 0, !dbg !24
  store float 0.000000e+00, ptr %arrayidx, align 4, !dbg !24, !DIAssignID !50
  ;; -- End modification
  call void @_Z3escP1v(ptr nonnull %g), !dbg !43
  ret void, !dbg !45
}

declare !dbg !64 dso_local void @_Z3escP1v(ptr) local_unnamed_addr
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduce.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !1, file: !1, line: 12, type: !8, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "g", scope: !7, file: !1, line: 13, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v", file: !1, line: 1, size: 128, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !13, identifier: "_ZTS1v")
!13 = !{!14, !19}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !12, file: !1, line: 2, baseType: !15, size: 128)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 128, elements: !17)
!16 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!17 = !{!18}
!18 = !DISubrange(count: 4)
!19 = !DISubprogram(name: "v", scope: !12, file: !1, line: 4, type: !20, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22, !16}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!23 = distinct !DIAssignID()
!24 = !DILocation(line: 0, scope: !7)
!25 = !DILocation(line: 13, column: 3, scope: !7)
!26 = !DILocalVariable(name: "this", arg: 1, scope: !27, type: !30, flags: DIFlagArtificial | DIFlagObjectPointer)
!27 = distinct !DISubprogram(name: "v", linkageName: "_ZN1vC2Ef", scope: !12, file: !1, line: 4, type: !20, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !19, retainedNodes: !28)
!28 = !{!26, !29}
!29 = !DILocalVariable(name: "d", arg: 2, scope: !27, file: !1, line: 4, type: !16)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!31 = distinct !DIAssignID()
!32 = !DILocation(line: 0, scope: !27, inlinedAt: !33)
!33 = distinct !DILocation(line: 13, column: 5, scope: !7)
!34 = distinct !DIAssignID()
!35 = distinct !DIAssignID()
!36 = distinct !DIAssignID()
!37 = !DILocation(line: 5, column: 19, scope: !38, inlinedAt: !33)
!38 = distinct !DILexicalBlock(scope: !27, file: !1, line: 4, column: 14)
!39 = distinct !DIAssignID()
!40 = !DILocation(line: 5, column: 12, scope: !38, inlinedAt: !33)
!41 = !DILocation(line: 5, column: 5, scope: !38, inlinedAt: !33)
!42 = !DILocation(line: 6, column: 5, scope: !38, inlinedAt: !33)
!43 = !DILocation(line: 14, column: 3, scope: !7)
!44 = !DILocation(line: 5, column: 17, scope: !38, inlinedAt: !33)
!45 = !DILocation(line: 15, column: 1, scope: !7)
!46 = distinct !DIAssignID()
!47 = !DILocation(line: 0, scope: !27)
!48 = distinct !DIAssignID()
!49 = distinct !DIAssignID()
!50 = distinct !DIAssignID()
!51 = !DILocation(line: 5, column: 19, scope: !38)
!52 = !DILocation(line: 5, column: 24, scope: !38)
!57 = !DILocation(line: 5, column: 12, scope: !38)
!58 = !DILocation(line: 5, column: 17, scope: !38)
!59 = !DILocation(line: 5, column: 5, scope: !38)
!60 = !DILocation(line: 5, column: 10, scope: !38)
!61 = !DILocation(line: 6, column: 5, scope: !38)
!62 = !DILocation(line: 6, column: 10, scope: !38)
!63 = !DILocation(line: 7, column: 3, scope: !27)
!64 = !DISubprogram(name: "esc", linkageName: "_Z3escP1v", scope: !1, file: !1, line: 10, type: !65, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!65 = !DISubroutineType(types: !66)
!66 = !{null, !30}
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
