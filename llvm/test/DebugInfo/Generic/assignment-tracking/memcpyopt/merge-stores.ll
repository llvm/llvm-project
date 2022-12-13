; RUN: opt %s -S -passes=memcpyopt -o - -experimental-assignment-tracking | FileCheck %s

;; $ cat test.cpp
;; struct v {
;;   float c[4];
;;
;;   v(float d) {
;;     c[0] = c[1] = c[2] = d;
;;     c[3] = 0.f;
;;   }
;; };
;;
;; void esc(v*);
;;
;; v f() {
;;   v g(0);
;;   esc(&g);
;; }
;; IR grabbed before memcpyopt in:
;; clang++ -Xclang -fexperimental-assignment-tracking -g -c -O2 test.cpp

;; Check that the memset that memcpyopt creates to merge 4 stores merges the
;; DIASsignIDs from the stores.

; CHECK: call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata ![[VAR:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata ![[ID:[0-9]+]], metadata ptr %arrayidx.i, metadata !DIExpression())
; CHECK: call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata ![[ID]], metadata ptr %arrayidx3.i, metadata !DIExpression())
; CHECK: call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata ![[ID]], metadata ptr %arrayidx5.i, metadata !DIExpression())
; CHECK: call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 96, 32), metadata ![[ID]], metadata ptr %arrayidx7.i, metadata !DIExpression())
; CHECK: call void @llvm.memset{{.*}}, !DIAssignID ![[ID]]

%struct.v = type { [4 x float] }

$_ZN1vC2Ef = comdat any

; Function Attrs: uwtable
define dso_local void @_Z1fv() local_unnamed_addr #0 !dbg !7 {
entry:
  %g = alloca %struct.v, align 4, !DIAssignID !23
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !23, metadata ptr %g, metadata !DIExpression()), !dbg !24
  call void @llvm.lifetime.start.p0i8(i64 16, ptr nonnull %g) #5, !dbg !25
  call void @llvm.dbg.assign(metadata i1 undef, metadata !26, metadata !DIExpression(), metadata !31, metadata ptr undef, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.assign(metadata i1 undef, metadata !29, metadata !DIExpression(), metadata !34, metadata ptr undef, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.assign(metadata ptr %g, metadata !26, metadata !DIExpression(), metadata !35, metadata ptr undef, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !29, metadata !DIExpression(), metadata !36, metadata ptr undef, metadata !DIExpression()), !dbg !32
  %arrayidx.i = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 2, !dbg !37
  store float 0.000000e+00, ptr %arrayidx.i, align 4, !dbg !39, !DIAssignID !44
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !44, metadata ptr %arrayidx.i, metadata !DIExpression()), !dbg !24
  %arrayidx3.i = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 1, !dbg !45
  store float 0.000000e+00, ptr %arrayidx3.i, align 4, !dbg !46, !DIAssignID !47
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !47, metadata ptr %arrayidx3.i, metadata !DIExpression()), !dbg !24
  %arrayidx5.i = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 0, !dbg !48
  store float 0.000000e+00, ptr %arrayidx5.i, align 4, !dbg !49, !DIAssignID !50
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !50, metadata ptr %arrayidx5.i, metadata !DIExpression()), !dbg !24
  %arrayidx7.i = getelementptr inbounds %struct.v, ptr %g, i64 0, i32 0, i64 3, !dbg !51
  store float 0.000000e+00, ptr %arrayidx7.i, align 4, !dbg !52, !DIAssignID !53
  call void @llvm.dbg.assign(metadata float 0.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 96, 32), metadata !53, metadata ptr %arrayidx7.i, metadata !DIExpression()), !dbg !24
  call void @_Z3escP1v(ptr nonnull %g), !dbg !54
  call void @llvm.lifetime.end.p0i8(i64 16, ptr nonnull %g) #5, !dbg !55
  ret void, !dbg !55
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN1vC2Ef(ptr %this, float %d) unnamed_addr #2 comdat align 2 !dbg !27 {
entry:
  call void @llvm.dbg.assign(metadata i1 undef, metadata !26, metadata !DIExpression(), metadata !56, metadata ptr undef, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.assign(metadata i1 undef, metadata !29, metadata !DIExpression(), metadata !58, metadata ptr undef, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.assign(metadata ptr %this, metadata !26, metadata !DIExpression(), metadata !59, metadata ptr undef, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.assign(metadata float %d, metadata !29, metadata !DIExpression(), metadata !60, metadata ptr undef, metadata !DIExpression()), !dbg !57
  %arrayidx = getelementptr inbounds %struct.v, ptr %this, i64 0, i32 0, i64 2, !dbg !61
  store float %d, ptr %arrayidx, align 4, !dbg !62
  %arrayidx3 = getelementptr inbounds %struct.v, ptr %this, i64 0, i32 0, i64 1, !dbg !63
  store float %d, ptr %arrayidx3, align 4, !dbg !64
  %arrayidx5 = getelementptr inbounds %struct.v, ptr %this, i64 0, i32 0, i64 0, !dbg !65
  store float %d, ptr %arrayidx5, align 4, !dbg !66
  %arrayidx7 = getelementptr inbounds %struct.v, ptr %this, i64 0, i32 0, i64 3, !dbg !67
  store float 0.000000e+00, ptr %arrayidx7, align 4, !dbg !68
  ret void, !dbg !69
}

declare !dbg !70 dso_local void @_Z3escP1v(ptr) local_unnamed_addr
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
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
!39 = !DILocation(line: 5, column: 24, scope: !38, inlinedAt: !33)
!44 = distinct !DIAssignID()
!45 = !DILocation(line: 5, column: 12, scope: !38, inlinedAt: !33)
!46 = !DILocation(line: 5, column: 17, scope: !38, inlinedAt: !33)
!47 = distinct !DIAssignID()
!48 = !DILocation(line: 5, column: 5, scope: !38, inlinedAt: !33)
!49 = !DILocation(line: 5, column: 10, scope: !38, inlinedAt: !33)
!50 = distinct !DIAssignID()
!51 = !DILocation(line: 6, column: 5, scope: !38, inlinedAt: !33)
!52 = !DILocation(line: 6, column: 10, scope: !38, inlinedAt: !33)
!53 = distinct !DIAssignID()
!54 = !DILocation(line: 14, column: 3, scope: !7)
!55 = !DILocation(line: 15, column: 1, scope: !7)
!56 = distinct !DIAssignID()
!57 = !DILocation(line: 0, scope: !27)
!58 = distinct !DIAssignID()
!59 = distinct !DIAssignID()
!60 = distinct !DIAssignID()
!61 = !DILocation(line: 5, column: 19, scope: !38)
!62 = !DILocation(line: 5, column: 24, scope: !38)
!63 = !DILocation(line: 5, column: 12, scope: !38)
!64 = !DILocation(line: 5, column: 17, scope: !38)
!65 = !DILocation(line: 5, column: 5, scope: !38)
!66 = !DILocation(line: 5, column: 10, scope: !38)
!67 = !DILocation(line: 6, column: 5, scope: !38)
!68 = !DILocation(line: 6, column: 10, scope: !38)
!69 = !DILocation(line: 7, column: 3, scope: !27)
!70 = !DISubprogram(name: "esc", linkageName: "_Z3escP1v", scope: !1, file: !1, line: 10, type: !71, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!71 = !DISubroutineType(types: !72)
!72 = !{null, !30}
