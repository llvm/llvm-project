; RUN: opt %s -S -passes=dse -o - | FileCheck %s --implicit-check-not="#dbg_"

;; Based on the test shorten.ll with some adjustments.
;;
;; $ cat test.cpp
;; void esc(char*);
;; void shortenBeginWholeFragment() {
;;   char local[80];                      //        bits    frag
;;   __builtin_memset(local + 8, 0, 72);  // local:  64-160 (64, 96)
;;   __builtin_memset(local + 4, 8, 64);  // local:  32-160 killed
;;   esc(local);
;; }
;; void shortenStart() {
;;   char local2[40];                 //          bits   frag
;;   __builtin_memset(local2, 0, 36); // local2:  0-160  (0, 160)
;;   __builtin_memset(local2, 8, 16); // local2:  0-128  (0, 128)
;;   esc(local2);
;; }
;; void shortenEndPartial() {
;;   char local3[80];                      //         bits    frag
;;   __builtin_memset(local3 + 8,  0, 8);  // local3:  64-128 (64, 96)
;;   __builtin_memset(local3 + 12, 8, 4);  // local3:  96-128 (96, 32)
;;   esc(local3);
;; }

;; The variables and intrinsics have been adjusted by hand to test
;; what happens when the variable doesn't fill the whole alloca, and
;; when offsets are encoded with both the address component of the dbg.assign
;; and the address modifying DIExpression.

;; The debug variables for 'local' and 'local3' are 20 bytes even though their
;; allocas are 80 bytes. The range annotations stop at byte 20 when a write
;; continues past the variable.

;; shortenBeginWholeFragment checks the path where the later store starts before
;; the store DSE shortens, so DSE removes bytes from the beginning.
;; shortenEndPartial checks the path where the later store starts inside the
;; store being shortened and DSE removes bytes from the end.

;; The first memset in shortenBeginWholeFragment writes 72 bytes starting at
;; byte 8. The second memset overwrites its first 60 bytes. DSE removes the
;; first 48 bytes, bytes 8 through 55, and keeps the final 24 bytes so the
;; shortened memset remains 16-byte aligned.
;;
;; The dbg.assign record describes the 12 bytes of 'local' starting at byte 8.
;; DW_OP_LLVM_fragment stores the starting bit followed by the size, so the
;; CHECK expects (64, 96). DSE removes all 12 bytes, so make sure it unlinks the
;; whole record and kills its address.

; CHECK: @_Z25shortenBeginWholeFragmentv
; CHECK:      #dbg_assign({{.*}}, ![[VAR:[0-9]+]], !DIExpression(), {{.*}}, ptr %local, !DIExpression(),
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR]], !DIExpression(DW_OP_LLVM_fragment, 64, 96), ![[UniqueID1:[0-9]+]], ptr poison, !DIExpression(DW_OP_plus_uconst, 4),

;; DSE will shorten the first store in shortenStart from [0, 160) bits to [128,
;; 160) bits. Variable 'local2' has been adjusted to be 160 bits.  Check we get
;; an unlinked dbg.assign covering the deleted bits that overlap the
;; dbg.assign's fragment (no fragment in this case, i.e. the whole variable):
;; [0, 128) (offset=0, size=128).

; CHECK: @_Z12shortenStartv
; CHECK:      #dbg_assign({{.*}}, ptr %local2, !DIExpression(),
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID2:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR2:[0-9]+]], !DIExpression(), ![[ID2]], ptr %local2, !DIExpression(),
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR2]], !DIExpression(DW_OP_LLVM_fragment, 0, 128), ![[UniqueID2:[0-9]+]], ptr poison, !DIExpression(),

;; The first memset in shortenEndPartial writes eight bytes starting at byte 8.
;; The second memset overwrites its final four bytes, bytes 12 through 15, so
;; DSE keeps the first four bytes and removes the final four.
;;
;; The dbg.assign record describes 12 bytes starting at byte 8. The four
;; removed bytes start at byte 12 of 'local3', which is bit 96, and are 32 bits
;; long. DW_OP_LLVM_fragment stores the starting bit followed by the size, so
;; the CHECK expects (96, 32). Make sure the starting bit is 96 so that we know
;; the fragment starts where the second memset starts.

; CHECK: @_Z17shortenEndPartialv
; CHECK:      #dbg_assign({{.*}}, ptr %local3, !DIExpression(),
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID3:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR3:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 64, 96), ![[ID3]], ptr %offset_4_bytes, !DIExpression(DW_OP_plus_uconst, 4),
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR3]], !DIExpression(DW_OP_LLVM_fragment, 96, 32), ![[UniqueID3:[0-9]+]], ptr poison, !DIExpression(DW_OP_plus_uconst, 4),

; CHECK-DAG: ![[ID]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID1]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID2]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID3]] = distinct !DIAssignID()

define dso_local void @_Z25shortenBeginWholeFragmentv() local_unnamed_addr #0 !dbg !7 {
entry:
  %local = alloca [80 x i8], align 16, !DIAssignID !16
  call void @llvm.dbg.assign(metadata i1 poison, metadata !11, metadata !DIExpression(), metadata !16, metadata ptr %local, metadata !DIExpression()), !dbg !17
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
  call void @llvm.dbg.assign(metadata i1 poison, metadata !33, metadata !DIExpression(), metadata !37, metadata ptr %local2, metadata !DIExpression()), !dbg !38
  %arraydecay = getelementptr inbounds [40 x i8], ptr %local2, i64 0, i64 0, !dbg !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(40) %local2, i8 0, i64 36, i1 false), !dbg !40, !DIAssignID !41
  call void @llvm.dbg.assign(metadata i8 0, metadata !33, metadata !DIExpression(), metadata !41, metadata ptr %local2, metadata !DIExpression()), !dbg !38
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %local2, i8 8, i64 16, i1 false), !dbg !42, !DIAssignID !43
  call void @_Z3escPi(ptr noundef nonnull %arraydecay), !dbg !44
  ret void, !dbg !45
}

;; shortenEndPartial uses alignment 4 for the first memset so DSE can shorten it
;; from eight bytes to four. With alignment 16 DSE leaves the memset alone, so
;; the test would not check removing bytes from the end.
define dso_local void @_Z17shortenEndPartialv() local_unnamed_addr #0 !dbg !46 {
entry:
  %local3 = alloca [80 x i8], align 16, !DIAssignID !50
  call void @llvm.dbg.assign(metadata i1 poison, metadata !48, metadata !DIExpression(), metadata !50, metadata ptr %local3, metadata !DIExpression()), !dbg !49
  %arraydecay = getelementptr inbounds [80 x i8], ptr %local3, i64 0, i64 0, !dbg !53
  %offset_4_bytes = getelementptr inbounds [80 x i8], ptr %local3, i64 0, i64 4, !dbg !53
  %offset_8_bytes = getelementptr inbounds [80 x i8], ptr %local3, i64 0, i64 8, !dbg !53
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 %offset_8_bytes, i8 0, i64 8, i1 false), !dbg !53, !DIAssignID !51
  call void @llvm.dbg.assign(metadata i8 0, metadata !48, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 96), metadata !51, metadata ptr %offset_4_bytes, metadata !DIExpression(DW_OP_plus_uconst, 4)), !dbg !49
  %offset_12_bytes = getelementptr inbounds [80 x i8], ptr %local3, i64 0, i64 12, !dbg !53
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 %offset_12_bytes, i8 8, i64 4, i1 false), !dbg !54, !DIAssignID !52
  call void @_Z3escPi(ptr noundef nonnull %arraydecay), !dbg !55
  ret void, !dbg !56
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
!7 = distinct !DISubprogram(name: "shortenBeginWholeFragment", linkageName: "_Z25shortenBeginWholeFragmentv", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
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
!46 = distinct !DISubprogram(name: "shortenEndPartial", linkageName: "_Z17shortenEndPartialv", scope: !1, file: !1, line: 14, type: !8, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !47)
!47 = !{!48}
!48 = !DILocalVariable(name: "local3", scope: !46, file: !1, line: 15, type: !12)
!49 = !DILocation(line: 0, scope: !46)
!50 = distinct !DIAssignID()
!51 = distinct !DIAssignID()
!52 = distinct !DIAssignID()
!53 = !DILocation(line: 16, column: 3, scope: !46)
!54 = !DILocation(line: 17, column: 3, scope: !46)
!55 = !DILocation(line: 18, column: 3, scope: !46)
!56 = !DILocation(line: 19, column: 1, scope: !46)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
