; RUN: opt %s -S -passes=dse -o - | FileCheck %s

;; $ cat test.cpp
;; void esc(int*);
;; void shortenEnd() {
;;   int local[20];
;;   __builtin_memset(local, 0, 6 * 4);
;;   __builtin_memset(local + 4, 8, 10 * 4);
;;   esc(local);
;; }
;; void shortenStart() {
;;   int local2[10];
;;   __builtin_memset(local2, 0, 10 * 4);
;;   __builtin_memset(local2, 8, 4 * 4);
;;   esc(local2);
;; }
;; IR grabbed before dse in:
;; clang++ -O2 -g -Xclang -fexperimental-assignment-tracking

;; DeadStoreElimination will shorten the first store in shortenEnd from [0,
;; 192) bits to [0, 128) bits. Check that we get an unlinked dbg.assign covering
;; the deleted bits [128, 192) (offset=128 size=64). It will shorten also the
;; first store in shortenStart from [0, 320) bits to [128, 320). Check that we
;; get an unlinked dbg.assign covering the deleted bits [0, 128) (offset=0
;; size=128).

; CHECK: @_Z10shortenEndv
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i8 0, metadata ![[VAR:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 192), metadata ![[ID:[0-9]+]], metadata ptr %local, metadata !DIExpression())
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i8 0, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64), metadata ![[UniqueID1:[0-9]+]], metadata ptr undef, metadata !DIExpression())

; CHECK: @_Z12shortenStartv
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID2:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i8 0, metadata ![[VAR2:[0-9]+]], metadata !DIExpression(), metadata ![[ID2]], metadata ptr %local2, metadata !DIExpression())
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i8 0, metadata ![[VAR2]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 128), metadata ![[UniqueID2:[0-9]+]], metadata ptr undef, metadata !DIExpression())

; CHECK-DAG: ![[ID]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID1]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID2]] = distinct !DIAssignID()

define dso_local void @_Z10shortenEndv() local_unnamed_addr #0 !dbg !7 {
entry:
  %local = alloca [20 x i32], align 16, !DIAssignID !16
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !16, metadata ptr %local, metadata !DIExpression()), !dbg !17
  call void @llvm.lifetime.start.p0i8(i64 80, ptr nonnull %local) #5, !dbg !18
  %arraydecay = getelementptr inbounds [20 x i32], ptr %local, i64 0, i64 0, !dbg !19
  call void @llvm.memset.p0i8.i64(ptr noundef nonnull align 16 dereferenceable(24) %local, i8 0, i64 24, i1 false), !dbg !19, !DIAssignID !20
  call void @llvm.dbg.assign(metadata i8 0, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 192), metadata !20, metadata ptr %local, metadata !DIExpression()), !dbg !17
  %add.ptr = getelementptr inbounds [20 x i32], ptr %local, i64 0, i64 4, !dbg !21
  call void @llvm.memset.p0i8.i64(ptr noundef nonnull align 16 dereferenceable(40) %add.ptr, i8 8, i64 40, i1 false), !dbg !22, !DIAssignID !23
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 320), metadata !23, metadata ptr %add.ptr, metadata !DIExpression()), !dbg !17
  call void @_Z3escPi(ptr noundef nonnull %arraydecay), !dbg !24
  call void @llvm.lifetime.end.p0i8(i64 80, ptr nonnull %local) #5, !dbg !25
  ret void, !dbg !25
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.memset.p0i8.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare !dbg !26 dso_local void @_Z3escPi(ptr noundef) local_unnamed_addr
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)

define dso_local void @_Z12shortenStartv() local_unnamed_addr #0 !dbg !31 {
entry:
  %local2 = alloca [10 x i32], align 16, !DIAssignID !37
  call void @llvm.dbg.assign(metadata i1 undef, metadata !33, metadata !DIExpression(), metadata !37, metadata ptr %local2, metadata !DIExpression()), !dbg !38
  call void @llvm.lifetime.start.p0i8(i64 40, ptr nonnull %local2) #5, !dbg !39
  %arraydecay = getelementptr inbounds [10 x i32], ptr %local2, i64 0, i64 0, !dbg !40
  call void @llvm.memset.p0i8.i64(ptr noundef nonnull align 16 dereferenceable(40) %local2, i8 0, i64 40, i1 false), !dbg !40, !DIAssignID !41
  call void @llvm.dbg.assign(metadata i8 0, metadata !33, metadata !DIExpression(), metadata !41, metadata ptr %local2, metadata !DIExpression()), !dbg !38
  call void @llvm.memset.p0i8.i64(ptr noundef nonnull align 16 dereferenceable(16) %local2, i8 8, i64 16, i1 false), !dbg !42, !DIAssignID !43
  call void @llvm.dbg.assign(metadata i1 undef, metadata !33, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 128), metadata !43, metadata ptr %local2, metadata !DIExpression()), !dbg !38
  call void @_Z3escPi(ptr noundef nonnull %arraydecay), !dbg !44
  call void @llvm.lifetime.end.p0i8(i64 40, ptr nonnull %local2) #5, !dbg !45
  ret void, !dbg !45
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "shortenEnd", linkageName: "_Z10shortenEndv", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 3, type: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 640, elements: !14)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 20)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 3, column: 3, scope: !7)
!19 = !DILocation(line: 4, column: 3, scope: !7)
!20 = distinct !DIAssignID()
!21 = !DILocation(line: 5, column: 26, scope: !7)
!22 = !DILocation(line: 5, column: 3, scope: !7)
!23 = distinct !DIAssignID()
!24 = !DILocation(line: 6, column: 3, scope: !7)
!25 = !DILocation(line: 7, column: 1, scope: !7)
!26 = !DISubprogram(name: "esc", linkageName: "_Z3escPi", scope: !1, file: !1, line: 1, type: !27, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !30)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !29}
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!30 = !{}
!31 = distinct !DISubprogram(name: "shortenStart", linkageName: "_Z12shortenStartv", scope: !1, file: !1, line: 8, type: !8, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !32)
!32 = !{!33}
!33 = !DILocalVariable(name: "local2", scope: !31, file: !1, line: 9, type: !34)
!34 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 320, elements: !35)
!35 = !{!36}
!36 = !DISubrange(count: 10)
!37 = distinct !DIAssignID()
!38 = !DILocation(line: 0, scope: !31)
!39 = !DILocation(line: 9, column: 3, scope: !31)
!40 = !DILocation(line: 10, column: 3, scope: !31)
!41 = distinct !DIAssignID()
!42 = !DILocation(line: 11, column: 3, scope: !31)
!43 = distinct !DIAssignID()
!44 = !DILocation(line: 12, column: 3, scope: !31)
!45 = !DILocation(line: 13, column: 1, scope: !31)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
