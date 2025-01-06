; RUN: opt -S < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=2 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we are preserving debug info in the vectorized code.

; CHECK-LABEL: define i32 @f(
; CHECK: for.body.lr.ph
; CHECK:   min.iters.check = icmp ult i64 {{.*}}, 2, !dbg !{{[0-9]+}}
; CHECK: vector.body
; CHECK:   index {{.*}}, !dbg ![[LOC1:[0-9]+]]
; CHECK:   getelementptr inbounds i32, ptr %a, {{.*}}, !dbg ![[LOC2:[0-9]+]]
; CHECK:   getelementptr inbounds i32, ptr %b, {{.*}}, !dbg ![[LOC1]]
; CHECK:   load <2 x i32>, ptr {{.*}}, !dbg ![[LOC1]]
; CHECK:   add <2 x i32> {{.*}}, !dbg ![[LOC1]]
; CHECK:   add nuw i64 %index, 2, !dbg ![[LOC1]]
; CHECK:   icmp eq i64 %index.next, %n.vec, !dbg ![[LOC1]]
; CHECK: middle.block
; CHECK:   call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %{{.*}}), !dbg ![[BR_LOC:[0-9]+]]
; CHECK: for.body
; CHECK: br i1{{.*}}, label %for.body,{{.*}}, !dbg ![[BR_LOC]],

define i32 @f(ptr nocapture %a, ptr %b, i32 %size) !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata ptr %a, metadata !13, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %size, metadata !14, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !21
  %cmp4 = icmp eq i32 %size, 0, !dbg !21
  br i1 %cmp4, label %for.end, label %for.body.lr.ph, !dbg !21

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !21

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %sum = phi i32 [ 0, %for.body.lr.ph ], [ %sum.next, %for.body ]
  %arrayidx.1 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv, !dbg !19
  %arrayidx.2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv, !dbg !22
  %l.1 = load i32, ptr %arrayidx.1, align 4, !dbg !22
  %l.2 = load i32, ptr %arrayidx.2, align 4, !dbg !22
  %add.1 = add i32 %l.1, %l.2
  %sum.next = add i32 %add.1, %sum, !dbg !22
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !22
  call void @llvm.dbg.value(metadata !{null}, metadata !16, metadata !DIExpression()), !dbg !22
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !22
  %exitcond = icmp ne i32 %lftr.wideiv, %size, !dbg !21
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge, !dbg !21

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %add.lcssa = phi i32 [ %sum.next, %for.body ]
  call void @llvm.dbg.value(metadata i32 %add.lcssa, metadata !15, metadata !DIExpression()), !dbg !22
  br label %for.end, !dbg !21

for.end:                                          ; preds = %entry, %for.cond.for.end_crit_edge
  %sum.0.lcssa = phi i32 [ %add.lcssa, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa, !dbg !26
}

define i32 @test_debug_loc_on_branch_in_loop(ptr noalias %src, ptr noalias %dst) {
; CHECK-LABEL: define i32 @test_debug_loc_on_branch_in_loop(
; CHECK-LABEL: vector.body:
; CHECK:        [[LOAD:%.+]] = load <2 x i32>, ptr {{.+}}, align 4
; CHECK-NEXT:   [[CMP:%.+]] = icmp eq <2 x i32> [[LOAD]], splat (i32 10)
; CHECK-NEXT:   [[XOR:%.+]] = xor <2 x i1> [[CMP:%.+]], splat (i1 true), !dbg [[LOC3:!.+]]
; CHECK-NEXT:   [[EXT:%.+]] = extractelement <2 x i1> [[XOR]], i32 0, !dbg [[LOC3]]
; CHECK-NEXT:   br i1 [[EXT]], label %pred.store.if, label %pred.store.continue
; CHECK-NOT:  !dbg
; CHECK-EMPTY:
; CHECK-NEXT: pred.store.if:
; CHECK-NEXT:   [[GEP:%.+]] = getelementptr inbounds i32, ptr %dst, i64 {{.+}}, !dbg [[LOC3]]
; CHECK-NEXT:   store i32 0, ptr [[GEP]], align 4, !dbg [[LOC3]]
; CHECK-NEXT:   br label %pred.store.continue, !dbg [[LOC3]]
; CHECK-EMPTY:
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %l = load i32, ptr %gep.src, align 4
  %cmp = icmp eq i32 %l, 10
  br i1 %cmp, label %loop.latch, label %if.then, !dbg !28

if.then:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 0, ptr %gep.dst, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:
  ret i32 0
}

define i32 @test_different_debug_loc_on_replicate_recipe(ptr noalias %src, ptr noalias %dst) {
; CHECK-LABEL: define i32 @test_different_debug_loc_on_replicate_recipe(
; CHECK-LABEL: vector.body:
; CHECK:        [[LOAD:%.+]] = load <2 x i32>, ptr {{.+}}, align 4
; CHECK-NEXT:   [[CMP:%.+]] = icmp eq <2 x i32> [[LOAD]], splat (i32 10)
; CHECK-NEXT:   [[XOR:%.+]] = xor <2 x i1> [[CMP:%.+]], splat (i1 true), !dbg [[LOC4:!.+]]
; CHECK-NEXT:   [[EXT:%.+]] = extractelement <2 x i1> [[XOR]], i32 0, !dbg [[LOC4]]
; CHECK-NEXT:   br i1 [[EXT]], label %pred.store.if, label %pred.store.continue
; CHECK-NOT:  !dbg
; CHECK-EMPTY:
; CHECK-NEXT: pred.store.if:
; CHECK-NEXT:   [[GEP:%.+]] = getelementptr inbounds i32, ptr %dst, i64 {{.+}}, !dbg [[LOC5:!.+]]
; CHECK-NEXT:   store i32 0, ptr [[GEP]], align 4, !dbg [[LOC5]]
; CHECK-NEXT:   br label %pred.store.continue, !dbg [[LOC4]]
; CHECK-EMPTY:
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %l = load i32, ptr %gep.src, align 4
  %cmp = icmp eq i32 %l, 10
  br i1 %cmp, label %loop.latch, label %if.then, !dbg !33

if.then:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv, !dbg !34
  store i32 0, ptr %gep.dst, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:
  ret i32 0
}

define void @test_misc(ptr nocapture %a, ptr noalias %b, i64 %size) !dbg !35 {
; CHECK-LABEL: define void @test_misc(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr %a, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr %b, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <2 x i32>, ptr [[TMP3]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = icmp uge <2 x i32> [[WIDE_LOAD]], splat (i32 10)
; CHECK-NEXT:    [[TMP5:%.*]] = select <2 x i1> [[TMP4]], <2 x i32> [[WIDE_LOAD]], <2 x i32> zeroinitializer, !dbg [[LOC6:![0-9]+]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[TMP2]], i32 0, !dbg [[LOC7:![0-9]+]]
; CHECK-NEXT:    store <2 x i32> [[TMP5]], ptr [[TMP6]], align 4, !dbg [[LOC7]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP7:%.*]] = icmp eq i64 [[INDEX_NEXT]],
; CHECK-NEXT:    br i1 [[TMP7]], label %middle.block, label %vector.body
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx.1 = getelementptr inbounds i32, ptr %a, i64 %iv
  %arrayidx.2 = getelementptr inbounds i32, ptr %b, i64 %iv
  %l.1 = load i32, ptr %arrayidx.1, align 4
  %c = icmp uge i32 %l.1, 10
  %sel = select i1 %c, i32 %l.1, i32 0, !dbg !37
  store i32 %sel, ptr %arrayidx.2, !dbg !38
  %iv.next = add i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, %size
  br i1 %exitcond, label %loop, label %exit

exit:
  ret void
}

; CHECK: ![[LOC2]] = !DILocation(line: 3
; CHECK: ![[BR_LOC]] = !DILocation(line: 5,
; CHECK: ![[LOC1]] = !DILocation(line: 6
; CHECK: [[LOC3]] = !DILocation(line: 137
; CHECK: [[LOC4]] = !DILocation(line: 210
; CHECK: [[LOC5]] = !DILocation(line: 320
; CHECK: [[LOC6]] = !DILocation(line: 430
; CHECK: [[LOC7]] = !DILocation(line: 540


declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !27}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (trunk 185038) (llvm/trunk 185097)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "-", directory: "/Volumes/Data/backedup/dev/os/llvm/debug")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 3, file: !5, scope: !6, type: !7, retainedNodes: !12)
!5 = !DIFile(filename: "<stdin>", directory: "/Volumes/Data/backedup/dev/os/llvm/debug")
!6 = !DIFile(filename: "<stdin>", directory: "/Volumes/Data/backedup/dev/os/llvm/debug")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !11}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!12 = !{!13, !14, !15, !16}
!13 = !DILocalVariable(name: "a", line: 3, arg: 1, scope: !4, file: !6, type: !10)
!14 = !DILocalVariable(name: "size", line: 3, arg: 2, scope: !4, file: !6, type: !11)
!15 = !DILocalVariable(name: "sum", line: 4, scope: !4, file: !6, type: !11)
!16 = !DILocalVariable(name: "i", line: 5, scope: !17, file: !6, type: !11)
!17 = distinct !DILexicalBlock(line: 5, column: 0, file: !5, scope: !4)
!18 = !{i32 2, !"Dwarf Version", i32 3}
!19 = !DILocation(line: 3, scope: !4)
!20 = !DILocation(line: 4, scope: !4)
!21 = !DILocation(line: 5, scope: !17)
!22 = !DILocation(line: 6, scope: !17)
!26 = !DILocation(line: 7, scope: !4)
!27 = !{i32 1, !"Debug Info Version", i32 3}
!28 = !DILocation(line: 137, column: 44, scope: !29)
!29 = distinct !DILexicalBlock(scope: !30, file: !5, line: 137, column: 2)
!30 = distinct !DISubprogram(name: "Place", scope: !5, file: !5, line: 135, scopeLine: 135, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = distinct !DISubprogram(name: "Place", scope: !5, file: !5, line: 135, scopeLine: 135, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!32 = distinct !DILexicalBlock(scope: !31, file: !5, line: 137, column: 2)
!33 = !DILocation(line: 210, column: 44, scope: !32)
!34 = !DILocation(line: 320, column: 44, scope: !32)
!35 = distinct !DISubprogram(name: "test_misc", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 3, file: !5, scope: !6, type: !7, retainedNodes: !12)
!36 = distinct !DILexicalBlock(scope: !35, file: !5, line: 137, column: 2)
!37 = !DILocation(line: 430, column: 44, scope: !36)
!38 = !DILocation(line: 540, column: 44, scope: !36)
