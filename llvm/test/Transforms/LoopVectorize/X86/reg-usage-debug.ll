; RUN: opt < %s -debug-only=loop-vectorize -loop-vectorize -mtriple=x86_64-unknown-linux -S 2>&1 | FileCheck %s
; REQUIRES: asserts

; Test that the register usage estimation is not affected by the presence of
; debug intrinsics.
;
; In the test below the values %0 and %r.08 are ended in the add instruction
; preceding the call to the intrinsic, and will be recorded against the index
; of the call instruction.  This means the debug intrinsic must be considered
; when erasing instructions from the list of open-intervals.
;
; Tests generated from following source (with and without -g):

; unsigned test(unsigned *a, unsigned n) {
;   unsigned i, r = 0;
;   for(i = 0; i < n; i++)
;     r += a[i];
;   return r;
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: LV: Checking a loop in "test_g"
; CHECK: LV(REG): Found max usage: 2

define i32 @test_g(i32* nocapture readonly %a, i32 %n) local_unnamed_addr !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %a, i64 0, metadata !12, metadata !16), !dbg !17
  tail call void @llvm.dbg.value(metadata i32 %n, i64 0, metadata !13, metadata !16), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !15, metadata !16), !dbg !19
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !14, metadata !16), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !15, metadata !16), !dbg !19
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !14, metadata !16), !dbg !20
  %cmp6 = icmp eq i32 %n, 0, !dbg !21
  br i1 %cmp6, label %for.end, label %for.body.preheader, !dbg !25

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64, !dbg !21
  br label %for.body, !dbg !27

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %r.08 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv, !dbg !27
  %0 = load i32, i32* %arrayidx, align 4, !dbg !27, !tbaa !28
  %add = add i32 %0, %r.08, !dbg !32
  tail call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !15, metadata !16), !dbg !19
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !33
  tail call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !15, metadata !16), !dbg !19
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count, !dbg !21
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !25, !llvm.loop !35

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end, !dbg !38

for.end:                                          ; preds = %for.end.loopexit, %entry
  %r.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.end.loopexit ]
  ret i32 %r.0.lcssa, !dbg !38
}

; CHECK: LV: Checking a loop in "test"
; CHECK: LV(REG): Found max usage: 2

define i32 @test(i32* nocapture readonly %a, i32 %n) local_unnamed_addr {
entry:
  %cmp6 = icmp eq i32 %n, 0
  br i1 %cmp6, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %r.08 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !28
  %add = add i32 %0, %r.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %r.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.end.loopexit ]
  ret i32 %r.0.lcssa
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test_g", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !11)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !9}
!9 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!11 = !{!12, !13, !14, !15}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !6, file: !1, line: 1, type: !10)
!13 = !DILocalVariable(name: "n", arg: 2, scope: !6, file: !1, line: 1, type: !9)
!14 = !DILocalVariable(name: "i", scope: !6, file: !1, line: 2, type: !9)
!15 = !DILocalVariable(name: "r", scope: !6, file: !1, line: 2, type: !9)
!16 = !DIExpression()
!17 = !DILocation(line: 1, column: 27, scope: !6)
!18 = !DILocation(line: 1, column: 39, scope: !6)
!19 = !DILocation(line: 2, column: 15, scope: !6)
!20 = !DILocation(line: 2, column: 12, scope: !6)
!21 = !DILocation(line: 3, column: 16, scope: !22)
!22 = !DILexicalBlockFile(scope: !23, file: !1, discriminator: 1)
!23 = distinct !DILexicalBlock(scope: !24, file: !1, line: 3, column: 3)
!24 = distinct !DILexicalBlock(scope: !6, file: !1, line: 3, column: 3)
!25 = !DILocation(line: 3, column: 3, scope: !26)
!26 = !DILexicalBlockFile(scope: !24, file: !1, discriminator: 1)
!27 = !DILocation(line: 4, column: 10, scope: !23)
!28 = !{!29, !29, i64 0}
!29 = !{!"int", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
!32 = !DILocation(line: 4, column: 7, scope: !23)
!33 = !DILocation(line: 3, column: 22, scope: !34)
!34 = !DILexicalBlockFile(scope: !23, file: !1, discriminator: 2)
!35 = distinct !{!35, !36, !37}
!36 = !DILocation(line: 3, column: 3, scope: !24)
!37 = !DILocation(line: 4, column: 13, scope: !24)
!38 = !DILocation(line: 5, column: 3, scope: !6)
