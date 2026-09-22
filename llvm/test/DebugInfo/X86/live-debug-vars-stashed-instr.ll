; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O2 -experimental-debug-variable-locations=true -stop-after=livedebugvalues %s -o - | FileCheck %s

; In instruction referencing mode LiveDebugVariables unlinks debug instructions
; during register allocation and records the SlotIndex to put them back at, in
; StashedDebugInstrs. Nothing tells it when the instruction at that index is
; erased in between, so here the recorded index is left naming one that has
; been erased, and emitDebugValues takes its "insert position disappeared"
; path, walking forwards through the slots for a new one. Note that this is
; the opposite direction to findInsertLocation, which scans backwards.
;
; Reduced from MultiSource/Applications/oggenc built at -O2 -g.

; CHECK-LABEL: name: accumulate_fit
; CHECK: DBG_VALUE 0, $noreg, ![[VAR:[0-9]+]], !DIExpression(), debug-location ![[LOC:[0-9]+]]

define fastcc i32 @accumulate_fit(ptr %flr, ptr %mdct, i32 %x0, i32 %x1, ptr %a, i32 %n, ptr %info, i64 %i.0141) !dbg !4 {
entry:
    #dbg_value(i64 0, !7, !DIExpression(), !9)
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
  %0 = load float, ptr %add.ptr, align 4
  %conv.i = fptosi float %0 to i32
  %arrayidx = getelementptr [4 x i8], ptr %mdct, i64 %i.01412
  %1 = load float, ptr %arrayidx, align 4
  %2 = load float, ptr %info, align 4
  %add = fadd float %1, %2
  %3 = tail call i32 @llvm.umin.i32(i32 %conv.i, i32 1)
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
  %conv25 = zext i32 %3 to i64
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

declare i32 @llvm.umin.i32(i32, i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "oggenc.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "accumulate_fit", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = distinct !DISubroutineType(types: !2)
!2 = !{}
!7 = !DILocalVariable(name: "xa", scope: !4, file: !1, line: 2, type: !8)
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!9 = !DILocation(line: 0, scope: !4)
