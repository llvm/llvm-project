; RUN: opt -S -loop-reduce %s | FileCheck %s

;; We (currently?) can't salvage an IV if the offset is wider than 64 bits.
;; Check we poison it instead.

; CHECK: #dbg_value(i[[#]] poison, ![[#]], !DIExpression(), ![[#]])

define i16 @main() {
entry:
  br label %for.cond29

for.cond29:                                       ; preds = %for.body32, %entry
  %il_1000.0 = phi i128 [ 0, %entry ], [ %inc72, %for.body32 ]
  %l_995.0 = phi i128 [ 4704496199548239085565, %entry ], [ %inc70, %for.body32 ]
    #dbg_value(i128 %l_995.0, !4, !DIExpression(), !9)
  %cmp30 = icmp slt i128 %il_1000.0, 0
  br i1 %cmp30, label %for.body32, label %for.cond.cleanup31

for.cond.cleanup31:                               ; preds = %for.cond29
  ret i16 0

for.body32:                                       ; preds = %for.cond29
  %inc70 = add i128 %l_995.0, 1
  %inc72 = add i128 %il_1000.0, 1
  br label %for.cond29
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "l_995", scope: !5, file: !1, line: 414, type: !7)
!5 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 397, type: !6, scopeLine: 398, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint128_t", file: !1, baseType: !8)
!8 = !DIBasicType(name: "unsigned __int128", size: 128, encoding: DW_ATE_unsigned)
!9 = !DILocation(line: 0, scope: !5)
