; RUN: opt %s -S -passes=dse -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators %s -S -passes=dse -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Based on the test shorten.ll with some adjustments.
;;
;; $ cat test.cpp
;; void esc(char*);
;; void shortenEnd() {
;;   char local[80];                      //        bits    frag
;;   __builtin_memset(local + 8,  0, 24); // local:  64-160 ( 64, 96)
;;   __builtin_memset(local + 16, 8, 40); // local: 128-160 (128, 32)
;;   esc(local);
;; }
;; void shortenStart() {
;;   char local2[40];                 //          bits   frag
;;   __builtin_memset(local2, 0, 40); // local2:  0-160  (0, 160)
;;   __builtin_memset(local2, 8, 16); // local2:  0-128  (0, 128)
;;   esc(local2);
;; }

;; The variables and intrinsics have been adjusted with by hand to test
;; what happens when the variable doesn't fill the whole alloca, and
;; when offsets are encoded with both the address component of the dbg.assign
;; and the address modifying DIExpression.

;; DeadStoreElimination will shorten the first store in shortenEnd from [64,
;; 192) bits to [64, 128) bits. Variable 'local' has been adjusted to be 160
;; bits large. Check that we get an unlinked dbg.assign covering the deleted
;; bits that overlap the dbg.assign's fagment: [128, 160) (offset=128 size=32).

; CHECK: @_Z10shortenEndv
; CHECK:      #dbg_assign({{.*}}, ptr %local, !DIExpression(),
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 64, 96), ![[ID:[0-9]+]], ptr %offset_4_bytes, !DIExpression(DW_OP_plus_uconst, 4),
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR]], !DIExpression(DW_OP_LLVM_fragment, 128, 32), ![[UniqueID1:[0-9]+]], ptr undef, !DIExpression({{.*}}),

;; DSE will shorten the first store in shortenStart from [0, 160) bits to [128,
;; 160) bits. Variable 'local2' has been adjusted to be 160 bits.  Check we get
;; an unlinked dbg.assign covering the deleted bits that overlap the
;; dbg.assign's fragment (no fragment in this case, i.e. the whole variable):
;; [0, 128) (offset=0, size=128).

; CHECK: @_Z12shortenStartv
; CHECK:      #dbg_assign({{.*}}, ptr %local2, !DIExpression(),
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID2:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR2:[0-9]+]], !DIExpression(), ![[ID2]], ptr %local2, !DIExpression(),
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR2]], !DIExpression(DW_OP_LLVM_fragment, 0, 128), ![[UniqueID2:[0-9]+]], ptr undef, !DIExpression(),

; CHECK-DAG: ![[ID]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID1]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID2]] = distinct !DIAssignID()

define dso_local void @_Z10shortenEndv() local_unnamed_addr #0 !dbg !7 {
entry:
  %local = alloca [80 x i8], align 16, !DIAssignID !16
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !16, metadata ptr %local, metadata !DIExpression()), !dbg !17
  %arraydecay = getelementptr inbounds [80 x i8], ptr %local, i64 0, i64 0, !dbg !19
  %offset_4_bytes = getelementptr inbounds [80 x i8], ptr %local, i64 0, i64 4, !dbg !21
  %offset_8_bytes = getelementptr inbounds [80 x i8], ptr %local, i64 0, i64 8, !dbg !21
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(24) %offset_8_bytes, i8 0, i64 72, i1 false), !dbg !19, !DIAssignID !20
  call void @llvm.dbg.assign(metadata i8 0, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 96), metadata !20, metadata ptr %offset_4_bytes, metadata !DIExpression(DW_OP_plus_uconst, 4)), !dbg !17
  %offset_16_bytes = getelementptr inbounds [80 x i8], ptr %local, i64 0, i64 4, !dbg !21
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(40) %offset_16_bytes, i8 8, i64 64, i1 false), !dbg !22, !DIAssignID !23
  call void @_Z3escPi(ptr noundef nonnull %arraydecay), !dbg !24
  ret void, !dbg !25
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare !dbg !26 dso_local void @_Z3escPi(ptr noundef) local_unnamed_addr

define dso_local void @_Z12shortenStartv() local_unnamed_addr #0 !dbg !31 {
entry:
  %local2 = alloca [40 x i8], align 16, !DIAssignID !37
  call void @llvm.dbg.assign(metadata i1 undef, metadata !33, metadata !DIExpression(), metadata !37, metadata ptr %local2, metadata !DIExpression()), !dbg !38
  %arraydecay = getelementptr inbounds [40 x i8], ptr %local2, i64 0, i64 0, !dbg !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(40) %local2, i8 0, i64 36, i1 false), !dbg !40, !DIAssignID !41
  call void @llvm.dbg.assign(metadata i8 0, metadata !33, metadata !DIExpression(), metadata !41, metadata ptr %local2, metadata !DIExpression()), !dbg !38
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %local2, i8 8, i64 16, i1 false), !dbg !42, !DIAssignID !43
  call void @_Z3escPi(ptr noundef nonnull %arraydecay), !dbg !44
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
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 160, elements: !14)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 5)
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
!34 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 160, elements: !35)
!35 = !{!36}
!36 = !DISubrange(count: 5)
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
