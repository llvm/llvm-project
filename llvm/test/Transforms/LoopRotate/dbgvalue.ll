; RUN: opt -S -passes=loop-rotate -verify-memoryssa < %s | FileCheck %s --implicit-check-not=dbg.value
; RUN: opt -S -passes=loop-rotate -verify-memoryssa < %s --try-experimental-debuginfo-iterators | FileCheck %s --implicit-check-not=dbg.value

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone


; This function rotates the exit conditon into the entry block, moving the
; dbg.values with it. Check that they resolve through the PHIs to the arguments
; only in the entry block. In the loop block, the dbg.values shift down below
; the calls and resolve to them. Then even more dbg.values are inserted on the
; newly produced PHIs at the start.

define i32 @tak(i32 %x, i32 %y, i32 %z) nounwind ssp !dbg !0 {
; CHECK-LABEL: define i32 @tak(
; CHECK: entry
; CHECK-NEXT: #dbg_value(i32 %x
; CHECK-NEXT: #dbg_value(i32 %y
; CHECK-NEXT: #dbg_value(i32 %z
; CHECK: if.then.lr.ph:
; CHECK: if.then:
; CHECK-NEXT: %z.tr4 = phi
; CHECK-NEXT: %y.tr3 = phi
; CHECK-NEXT: %x.tr2 = phi
; CHECK-NEXT: #dbg_value(i32 %z.tr4
; CHECK-NEXT: #dbg_value(i32 %y.tr3
; CHECK-NEXT: #dbg_value(i32 %x.tr2
; CHECK:      %call = tail call i32 @tak(i32
; CHECK:      %call9 = tail call i32 @tak(i32
; CHECK:      %call14 = tail call i32 @tak(i32
; CHECK-NEXT: #dbg_value(i32 %call
; CHECK-NEXT: #dbg_value(i32 %call9
; CHECK-NEXT: #dbg_value(i32 %call14
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then, %entry
  %x.tr = phi i32 [ %x, %entry ], [ %call, %if.then ]
  %y.tr = phi i32 [ %y, %entry ], [ %call9, %if.then ]
  %z.tr = phi i32 [ %z, %entry ], [ %call14, %if.then ]
  tail call void @llvm.dbg.value(metadata i32 %x.tr, metadata !6, metadata !DIExpression()), !dbg !7
  tail call void @llvm.dbg.value(metadata i32 %y.tr, metadata !8, metadata !DIExpression()), !dbg !9
  tail call void @llvm.dbg.value(metadata i32 %z.tr, metadata !10, metadata !DIExpression()), !dbg !11
  %cmp = icmp slt i32 %y.tr, %x.tr, !dbg !12
  br i1 %cmp, label %if.then, label %if.end, !dbg !12

if.then:                                          ; preds = %tailrecurse
  %sub = sub nsw i32 %x.tr, 1, !dbg !14
  %call = tail call i32 @tak(i32 %sub, i32 %y.tr, i32 %z.tr), !dbg !14
  %sub6 = sub nsw i32 %y.tr, 1, !dbg !14
  %call9 = tail call i32 @tak(i32 %sub6, i32 %z.tr, i32 %x.tr), !dbg !14
  %sub11 = sub nsw i32 %z.tr, 1, !dbg !14
  %call14 = tail call i32 @tak(i32 %sub11, i32 %x.tr, i32 %y.tr), !dbg !14
  br label %tailrecurse

if.end:                                           ; preds = %tailrecurse
  br label %return, !dbg !16

return:                                           ; preds = %if.end
  ret i32 %z.tr, !dbg !17
}

; Repeat of the tak function, with only one DILocalVariable, checking that we
; don't insert duplicate debug intrinsics. The initial duplicates are preserved.
; FIXME: this test checks for the de-duplication behaviour that loop-rotate
; has today, however it might not be correct. In the if.then block the preserved
; dbg.value is for %x -- should not the _last_dbg.value, for %z, have been
; preserved?
define i32 @tak_dup(i32 %x, i32 %y, i32 %z) nounwind ssp !dbg !50 {
; CHECK-LABEL: define i32 @tak_dup(
; CHECK: entry
; CHECK-NEXT: #dbg_value(i32 %x
; CHECK-NEXT: #dbg_value(i32 %y
; CHECK-NEXT: #dbg_value(i32 %z
; CHECK: if.then.lr.ph:
; CHECK: if.then:
; CHECK-NEXT: %z.tr4 = phi
; CHECK-NEXT: %y.tr3 = phi
; CHECK-NEXT: %x.tr2 = phi
; CHECK-NEXT: #dbg_value(i32 %x.tr2
; CHECK:      %call = tail call i32 @tak(i32
; CHECK:      %call9 = tail call i32 @tak(i32
; CHECK:      %call14 = tail call i32 @tak(i32
; CHECK-NEXT: #dbg_value(i32 %call14
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then, %entry
  %x.tr = phi i32 [ %x, %entry ], [ %call, %if.then ]
  %y.tr = phi i32 [ %y, %entry ], [ %call9, %if.then ]
  %z.tr = phi i32 [ %z, %entry ], [ %call14, %if.then ]
  tail call void @llvm.dbg.value(metadata i32 %x.tr, metadata !60, metadata !DIExpression()), !dbg !61
  tail call void @llvm.dbg.value(metadata i32 %y.tr, metadata !60, metadata !DIExpression()), !dbg !61
  tail call void @llvm.dbg.value(metadata i32 %z.tr, metadata !60, metadata !DIExpression()), !dbg !61
  %cmp = icmp slt i32 %y.tr, %x.tr, !dbg !62
  br i1 %cmp, label %if.then, label %if.end, !dbg !62

if.then:                                          ; preds = %tailrecurse
  %sub = sub nsw i32 %x.tr, 1, !dbg !64
  %call = tail call i32 @tak(i32 %sub, i32 %y.tr, i32 %z.tr), !dbg !64
  %sub6 = sub nsw i32 %y.tr, 1, !dbg !64
  %call9 = tail call i32 @tak(i32 %sub6, i32 %z.tr, i32 %x.tr), !dbg !64
  %sub11 = sub nsw i32 %z.tr, 1, !dbg !64
  %call14 = tail call i32 @tak(i32 %sub11, i32 %x.tr, i32 %y.tr), !dbg !64
  br label %tailrecurse

if.end:                                           ; preds = %tailrecurse
  br label %return, !dbg !66

return:                                           ; preds = %if.end
  ret i32 %z.tr, !dbg !67
}

; Check that the dbg.values move up to being immediately below the PHIs,
; using their Values. However once we exit the loop, the x and y values
; become irrelevant and undef, only z gets an LCSSA PHI to preserve it.
;
; Note that while the icmp is initially undominated by any dbg.value and thus
; shouldn't get a variable location, the first iteration is peeled off into the
; entry block. It's then safe to have it dominated by subsequent dbg.values as
; every path to the icmp is preceeded by a dbg.value.
;
; FIXME: could we choose to preserve more information about the loop, x and y
; might not be live out of the loop, but they might still be dominated by a
; describable Value.

define i32 @tak2(i32 %x, i32 %y, i32 %z) nounwind ssp !dbg !21 {
; CHECK-LABEL: define i32 @tak2(
; CHECK: if.then:
; CHECK-NEXT: %z.tr4 = phi i32
; CHECK-NEXT: %y.tr3 = phi i32
; CHECK-NEXT: %x.tr2 = phi i32
; CHECK-NEXT: #dbg_value(i32 %x.tr2
; CHECK-NEXT: #dbg_value(i32 %y.tr3
; CHECK-NEXT: #dbg_value(i32 %z.tr4
; CHECK:      tail call i32 @tak(i32
; CHECK:      tail call i32 @tak(i32
; CHECK:      tail call i32 @tak(i32
; CHECK: if.end:
; CHECK-NEXT: z.tr.lcssa = phi i32
; CHECK-NEXT: #dbg_value(i32 undef
; CHECK-NEXT: #dbg_value(i32 undef
; CHECK-NEXT: #dbg_value(i32 %z.tr.lcssa
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then, %entry
  %x.tr = phi i32 [ %x, %entry ], [ %call, %if.then ]
  %y.tr = phi i32 [ %y, %entry ], [ %call9, %if.then ]
  %z.tr = phi i32 [ %z, %entry ], [ %call14, %if.then ]
  %cmp = icmp slt i32 %y.tr, %x.tr, !dbg !22
  br i1 %cmp, label %if.then, label %if.end, !dbg !22

if.then:                                          ; preds = %tailrecurse
  tail call void @llvm.dbg.value(metadata i32 %x.tr, metadata !36, metadata !DIExpression()), !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %y.tr, metadata !38, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata i32 %z.tr, metadata !40, metadata !DIExpression()), !dbg !41
  %sub = sub nsw i32 %x.tr, 1, !dbg !24
  %call = tail call i32 @tak(i32 %sub, i32 %y.tr, i32 %z.tr), !dbg !24
  %sub6 = sub nsw i32 %y.tr, 1, !dbg !24
  %call9 = tail call i32 @tak(i32 %sub6, i32 %z.tr, i32 %x.tr), !dbg !24
  %sub11 = sub nsw i32 %z.tr, 1, !dbg !24
  %call14 = tail call i32 @tak(i32 %sub11, i32 %x.tr, i32 %y.tr), !dbg !24
  br label %tailrecurse

if.end:                                           ; preds = %tailrecurse
  tail call void @llvm.dbg.value(metadata i32 %x.tr, metadata !36, metadata !DIExpression()), !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %y.tr, metadata !38, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata i32 %z.tr, metadata !40, metadata !DIExpression()), !dbg !41
  br label %return, !dbg !26

return:                                           ; preds = %if.end
  ret i32 %z.tr, !dbg !27
}

@channelColumns = external global i64
@horzPlane = external global ptr, align 8

define void @FindFreeHorzSeg(i64 %startCol, i64 %row, ptr %rowStart) {
; Ensure that the loop increment basic block is rotated into the tail of the
; body, even though it contains a debug intrinsic call.
; CHECK-LABEL: define void @FindFreeHorzSeg(
; CHECK: %dec = add
; CHECK-NEXT: #dbg_value
; CHECK: %cmp = icmp
; CHECK: br i1 %cmp
; CHECK: phi i64 [ %{{[^,]*}}, %{{[^,]*}} ]
; CHECK-NEXT: br label %for.end


entry:
  br label %for.cond

for.cond:
  %i.0 = phi i64 [ %startCol, %entry ], [ %dec, %for.inc ]
  %cmp = icmp eq i64 %i.0, 0
  br i1 %cmp, label %for.end, label %for.body

for.body:
  %0 = load i64, ptr @channelColumns, align 8
  %mul = mul i64 %0, %row
  %add = add i64 %mul, %i.0
  %1 = load ptr, ptr @horzPlane, align 8
  %arrayidx = getelementptr inbounds i8, ptr %1, i64 %add
  %2 = load i8, ptr %arrayidx, align 1
  %tobool = icmp eq i8 %2, 0
  br i1 %tobool, label %for.inc, label %for.end

for.inc:
  %dec = add i64 %i.0, -1
  tail call void @llvm.dbg.value(metadata i64 %dec, metadata !DILocalVariable(scope: !0), metadata !DIExpression()), !dbg !DILocation(scope: !0)
  br label %for.cond

for.end:
  %add1 = add i64 %i.0, 1
  store i64 %add1, ptr %rowStart, align 8
  ret void
}

; Test that dbg.value intrinsincs adjacent to the `icmp slt i32 0, 0` get
; rotated as expected. The icmp is loop-invariant and so gets hoisted to the
; preheader via a different code path. This is more difficult for DbgVariableRecord
; debug-info records to handle, because they have to get detached and moved
; somewhere else during rotation.
define void @invariant_hoist() !dbg !70 {
; CHECK-LABEL: define void @invariant_hoist()
; CHECK: entry:
; CHECK-NEXT: br label %L0.preheader
; CHECK: L0.preheader:
; CHECK-NEXT: #dbg_value(i32 0,
; CHECK-NEXT: %cmp = icmp slt i32 0, 0,
; CHECK: L1.preheader:
; CHECK-NEXT: %spec.select3 = phi i32
; CHECK-NEXT: %k.02 = phi i32
; CHECK-NEXT: #dbg_value(i32 %k.02,
; CHECK: L0.latch:
; CHECK-NEXT: #dbg_value(i32 %spec.select3,
entry:
  br label %L0.preheader, !dbg !77

L0.preheader:
  br label %L0, !dbg !77

L0:
  %k.0 = phi i32 [ 0, %L0.preheader ], [ %spec.select, %L0.latch ]
  call void @llvm.dbg.value(metadata i32 %k.0, metadata !80, metadata !DIExpression()), !dbg !77
  %cmp = icmp slt i32 0, 0, !dbg !77
  %inc = zext i1 %cmp to i32, !dbg !77
  %spec.select = add nsw i32 %k.0, %inc, !dbg !77
  %tobool3.not = icmp eq i32 %spec.select, 0, !dbg !77
  br i1 %tobool3.not, label %L0.preheader, label %L1.preheader, !dbg !77

L1.preheader:
  %tobool8.not = icmp eq i32 %k.0, 0, !dbg !77
  br label %L1, !dbg !77

L1:
  br i1 %tobool8.not, label %L1.latch, label %L0.latch, !dbg !77

L1.latch:
  br i1 false, label %L1, label %L0.latch, !dbg !77

L0.latch:
  br label %L0, !dbg !77
}

!llvm.module.flags = !{!20}
!llvm.dbg.cu = !{!2}

!0 = distinct !DISubprogram(name: "tak", line: 32, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !18, scope: !1, type: !3)
!1 = !DIFile(filename: "/Volumes/Lalgate/cj/llvm/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame/recursive.c", directory: "/Volumes/Lalgate/cj/D/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 125492)", isOptimized: true, emissionKind: FullDebug, file: !18)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "x", line: 32, arg: 1, scope: !0, file: !1, type: !5)
!7 = !DILocation(line: 32, column: 13, scope: !0)
!8 = !DILocalVariable(name: "y", line: 32, arg: 2, scope: !0, file: !1, type: !5)
!9 = !DILocation(line: 32, column: 20, scope: !0)
!10 = !DILocalVariable(name: "z", line: 32, arg: 3, scope: !0, file: !1, type: !5)
!11 = !DILocation(line: 32, column: 27, scope: !0)
!12 = !DILocation(line: 33, column: 3, scope: !13)
!13 = distinct !DILexicalBlock(line: 32, column: 30, file: !18, scope: !0)
!14 = !DILocation(line: 34, column: 5, scope: !15)
!15 = distinct !DILexicalBlock(line: 33, column: 14, file: !18, scope: !13)
!16 = !DILocation(line: 36, column: 3, scope: !13)
!17 = !DILocation(line: 37, column: 1, scope: !13)
!18 = !DIFile(filename: "/Volumes/Lalgate/cj/llvm/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame/recursive.c", directory: "/Volumes/Lalgate/cj/D/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame")
!20 = !{i32 1, !"Debug Info Version", i32 3}
!21 = distinct !DISubprogram(name: "tak", line: 32, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !18, scope: !1, type: !3)
!22 = !DILocation(line: 33, column: 3, scope: !23)
!23 = distinct !DILexicalBlock(line: 32, column: 30, file: !18, scope: !21)
!24 = !DILocation(line: 34, column: 5, scope: !25)
!25 = distinct !DILexicalBlock(line: 33, column: 14, file: !18, scope: !23)
!26 = !DILocation(line: 36, column: 3, scope: !23)
!27 = !DILocation(line: 37, column: 1, scope: !23)
!36 = !DILocalVariable(name: "x", line: 32, arg: 1, scope: !21, file: !1, type: !5)
!37 = !DILocation(line: 32, column: 13, scope: !21)
!38 = !DILocalVariable(name: "y", line: 32, arg: 2, scope: !21, file: !1, type: !5)
!39 = !DILocation(line: 32, column: 20, scope: !21)
!40 = !DILocalVariable(name: "z", line: 32, arg: 3, scope: !21, file: !1, type: !5)
!41 = !DILocation(line: 32, column: 27, scope: !21)
!50 = distinct !DISubprogram(name: "tak_dup", line: 32, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !18, scope: !1, type: !3)
!57 = !DILocation(line: 32, column: 13, scope: !50)
!59 = !DILocation(line: 32, column: 20, scope: !50)
!60 = !DILocalVariable(name: "z", line: 32, arg: 3, scope: !50, file: !1, type: !5)
!61 = !DILocation(line: 32, column: 27, scope: !50)
!62 = !DILocation(line: 33, column: 3, scope: !63)
!63 = distinct !DILexicalBlock(line: 32, column: 30, file: !18, scope: !50)
!64 = !DILocation(line: 34, column: 5, scope: !65)
!65 = distinct !DILexicalBlock(line: 33, column: 14, file: !18, scope: !63)
!66 = !DILocation(line: 36, column: 3, scope: !63)
!67 = !DILocation(line: 37, column: 1, scope: !63)
!70 = distinct !DISubprogram(name: "invariant_hoist", line: 32, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !18, scope: !1, type: !3)
!77 = !DILocation(line: 32, column: 13, scope: !70)
!79 = !DILocation(line: 32, column: 20, scope: !70)
!80 = !DILocalVariable(name: "z", line: 32, arg: 3, scope: !70, file: !1, type: !5)
!81 = !DILocation(line: 32, column: 27, scope: !70)
!82 = !DILocation(line: 33, column: 3, scope: !83)
!83 = distinct !DILexicalBlock(line: 32, column: 30, file: !18, scope: !70)
!84 = !DILocation(line: 34, column: 5, scope: !85)
!85 = distinct !DILexicalBlock(line: 33, column: 14, file: !18, scope: !83)
!86 = !DILocation(line: 36, column: 3, scope: !83)
!87 = !DILocation(line: 37, column: 1, scope: !83)
