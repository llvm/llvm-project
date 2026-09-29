; RUN: llc -mtriple i386 %s -stop-after=livedebugvalues -o - | FileCheck %s

; There should be multiple debug values for this variable after regalloc. The
; value has been spilled, but we shouldn't lose track of the location because
; of this.

; CHECK-COUNT-4: DBG_VALUE $ebp, 0, !6, !DIExpression(DW_OP_constu, 16, DW_OP_minus), debug-location !10

define void @main(i32 %call, i32 %xor.i, i1 %tobool4.not, i32 %.pre) #0 !dbg !4 {
entry:
  %tobool1.not = icmp ne i32 %call, 0
  %spec.select = zext i1 %tobool1.not to i32
  br label %for.body5

for.cond.loopexit.loopexit:                       ; preds = %for.body5
    #dbg_value(i32 %spec.select, !6, !DIExpression(), !10)
  %tobool.not.i53 = icmp eq i32 %spec.select, 0
  br i1 %tobool.not.i53, label %transparent_crc.exit57, label %if.then.i54

for.body5:                                        ; preds = %for.body5, %entry
  %0 = phi i32 [ 0, %entry ], [ %xor1.i40.i, %for.body5 ]
  %xor6.i = xor i32 %.pre, %0
  %shr7.i = ashr i32 %xor6.i, 1
  %xor17.i = xor i32 %shr7.i, %call
  %shr18.i = ashr i32 %xor17.i, 1
  %xor.i.i = xor i32 %shr18.i, %xor.i
  %arrayidx.i.i = getelementptr [0 x i32], ptr null, i32 0, i32 %xor.i.i
  %xor1.i40.i = xor i32 %xor.i.i, %call
  br i1 %tobool4.not, label %for.cond.loopexit.loopexit, label %for.body5

if.then.i54:                                      ; preds = %for.cond.loopexit.loopexit
  store i64 0, ptr null, align 4
  br label %transparent_crc.exit57

transparent_crc.exit57:                           ; preds = %if.then.i54, %for.cond.loopexit.loopexit
  ret void
}

attributes #0 = { "frame-pointer"="all" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.0.0git.prerel", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "xx.c", directory: "/path", checksumkind: CSK_MD5, checksum: "c4b2fc62bca9171ad484c91fb78b8842")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 20, type: !5, scopeLine: 20, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!6 = !DILocalVariable(name: "flag", arg: 2, scope: !7, file: !1, line: 8, type: !9)
!7 = distinct !DISubprogram(name: "transparent_crc", scope: !1, file: !1, line: 8, type: !8, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = distinct !DISubroutineType(types: !2)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !7, inlinedAt: !11)
!11 = distinct !DILocation(line: 28, column: 3, scope: !4)
