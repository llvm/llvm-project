; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O2 -experimental-debug-variable-locations=true -stop-after=livedebugvalues %s -o - | FileCheck %s

; LiveDebugVariables records SlotIndexes during register allocation and uses
; them afterwards to re-insert debug instructions, but nothing tells it when an
; instruction is erased in between, so a recorded index can be left naming an
; instruction that no longer exists. Each function below reaches one such case.
; The debug instructions still come out in the right place, so these tests pin
; down that currently correct output.
;
; See also DebugInfo/AArch64/live-debug-vars-stale-slotindexes.ll.

; The SlotIndex of a DBG_LABEL is recorded in UserLabel::loc, and loc is left
; naming an erased instruction. insertDebugLabel goes through
; findInsertLocation, which scans backwards from the recorded index for a
; surviving instruction to insert after.
;
; Reduced from MultiSource/Benchmarks/Prolangs-C/cdecl built at -O2 -g.

; CHECK-LABEL: name: yyparse
; CHECK: DBG_LABEL !{{[0-9]+}}, debug-location !{{[0-9]+}}

define i32 @yyparse(ptr %add.ptr132, i16 %arg, i16 %arg1, i1 %cmp.not, i1 %cmp8) !dbg !4 {
entry:
    #dbg_label(!7, !8)
  br label %if.end

if.then:                                          ; preds = %yystack.backedge
  ret i32 0

if.end:                                           ; preds = %yystack.backedge, %entry
  %yystate.01028 = phi i16 [ 0, %entry ], [ %arg, %yystack.backedge ]
  store i16 %yystate.01028, ptr null, align 2
  br i1 %cmp8, label %if.then10, label %if.end15

if.then10:                                        ; preds = %if.end
  %call = tail call i32 null()
  br label %if.end15

if.end15:                                         ; preds = %if.then10, %if.end
  tail call void @llvm.memcpy.p0.p0.i64(ptr null, ptr %add.ptr132, i64 1, i1 false)
  switch i16 %arg1, label %yystack.backedge [
    i16 2, label %sw.bb162
    i16 4, label %sw.bb164
  ]

yystack.backedge:                                 ; preds = %if.end15
  br i1 %cmp.not, label %if.end, label %if.then

sw.bb162:                                         ; preds = %if.end15
  ret i32 0

sw.bb164:                                         ; preds = %if.end15
  ret i32 0
}

; In instruction referencing mode, debug instructions are unlinked during
; register allocation and the SlotIndex to put them back at is recorded in
; StashedDebugInstrs. Here that index is left naming an erased instruction, and
; emitDebugValues takes its "insert position disappeared" path, walking
; forwards through the slots for a new one. Note that this is the opposite
; direction to findInsertLocation, which scans backwards.
;
; Reduced from MultiSource/Applications/oggenc built at -O2 -g.

; CHECK-LABEL: name: accumulate_fit
; CHECK: DBG_VALUE 0, $noreg, !{{[0-9]+}}, !DIExpression(), debug-location !{{[0-9]+}}

define fastcc i32 @accumulate_fit(ptr %flr, ptr %mdct, i32 %x0, i32 %x1, ptr %a, i32 %n, ptr %info, i64 %i.0141) !dbg !13 {
entry:
    #dbg_value(i64 0, !16, !DIExpression(), !18)
  %conv = sext i32 %x0 to i64
  br label %for.body

for.body:                                         ; preds = %if.end37, %entry
  %y2b.0151 = phi i64 [ 0, %entry ], [ %y2b.1, %if.end37 ]
  %x2b.0150 = phi i64 [ 0, %entry ], [ %x2b.1, %if.end37 ]
  %yb.0149 = phi i64 [ 0, %entry ], [ %yb.1, %if.end37 ]
  %xb.0148 = phi i64 [ 0, %entry ], [ %xb.1, %if.end37 ]
  %na.0147 = phi i64 [ 0, %entry ], [ %na.1, %if.end37 ]
  %xya.0146 = phi i64 [ 0, %entry ], [ %xya.1, %if.end37 ]
  %y2a.0145 = phi i64 [ 0, %entry ], [ %y2a.1, %if.end37 ]
  %x2a.0144 = phi i64 [ 0, %entry ], [ %x2a.1, %if.end37 ]
  %i.01412 = phi i64 [ %conv, %entry ], [ 0, %if.end37 ]
  %add.ptr = getelementptr [4 x i8], ptr %flr, i64 %i.01412
  %i = load float, ptr %add.ptr, align 4
  %conv.i = fptosi float %i to i32
  %arrayidx = getelementptr [4 x i8], ptr %mdct, i64 %i.01412
  %i1 = load float, ptr %arrayidx, align 4
  %i2 = load float, ptr %info, align 4
  %add = fadd float %i1, %i2
  %i3 = tail call i32 @llvm.umin.i32(i32 %conv.i, i32 1)
  %mul273 = mul i64 %i.0141, %i.0141
  %cmp11 = fcmp ult float %add, 0.000000e+00
  br i1 %cmp11, label %if.else, label %if.then13

if.then13:                                        ; preds = %for.body
  %add17 = or i64 %x2a.0144, %mul273
  %add20 = or i64 %y2a.0145, 1
  %add23 = or i64 1, %xya.0146
  %inc = or i64 %na.0147, 1
  br label %if.end37

if.else:                                          ; preds = %for.body
  %add24 = or i64 %xb.0148, %i.0141
  %conv25 = zext i32 %i3 to i64
  %add26 = or i64 %yb.0149, %conv25
  %add28 = or i64 %x2b.0150, 1
  %add31 = or i64 %y2b.0151, %i.0141
  br label %if.end37

if.end37:                                         ; preds = %if.else, %if.then13
  %x2a.1 = phi i64 [ %add17, %if.then13 ], [ %x2a.0144, %if.else ]
  %y2a.1 = phi i64 [ %add20, %if.then13 ], [ %y2a.0145, %if.else ]
  %xya.1 = phi i64 [ %add23, %if.then13 ], [ %xya.0146, %if.else ]
  %na.1 = phi i64 [ %inc, %if.then13 ], [ %na.0147, %if.else ]
  %xb.1 = phi i64 [ %xb.0148, %if.then13 ], [ %add24, %if.else ]
  %yb.1 = phi i64 [ %yb.0149, %if.then13 ], [ %add26, %if.else ]
  %x2b.1 = phi i64 [ %x2b.0150, %if.then13 ], [ %add28, %if.else ]
  %y2b.1 = phi i64 [ %y2b.0151, %if.then13 ], [ %add31, %if.else ]
  br label %for.body
}

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "cdecl.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "yyparse", scope: !5, file: !5, line: 1022, type: !6, scopeLine: 1022, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2, keyInstructions: true)
!5 = !DIFile(filename: "cdgram.y", directory: "/")
!6 = !DISubroutineType(types: !2)
!7 = !DILabel(scope: !4, name: "yystack", file: !5, line: 1038, column: 2)
!8 = !DILocation(line: 1038, column: 2, scope: !4)
!9 = distinct !DICompileUnit(language: DW_LANG_C11, file: !10, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!10 = !DIFile(filename: "oggenc.c", directory: "/")
!13 = distinct !DISubprogram(name: "accumulate_fit", scope: !10, file: !10, line: 1, type: !15, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !9)
!15 = distinct !DISubroutineType(types: !11)
!11 = !{}
!16 = !DILocalVariable(name: "xa", scope: !13, file: !10, line: 2, type: !17)
!17 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!18 = !DILocation(line: 0, scope: !13)
