; RUN: opt -passes=loop-unroll -unroll-runtime -unroll-runtime-epilog -S %s | FileCheck %s
; RUN: opt -passes=loop-unroll -unroll-runtime -unroll-runtime-epilog -S %s --try-experimental-debuginfo-iterators | FileCheck %s

; Test that epilogue is tagged with the same debug information as original loop body rather than original loop exit.

; CHECK: for.body.i:
; CHECK:   br i1 {{.*}}, label %lee1.exit.loopexit.unr-lcssa.loopexit, label %for.body.i, !dbg ![[LOOP_LOC:[0-9]+]]
; CHECK: lee1.exit.loopexit.unr-lcssa.loopexit:
; CHECK:   br label %lee1.exit.loopexit.unr-lcssa, !dbg ![[LOOP_LOC]]
; CHECK: lee1.exit.loopexit.unr-lcssa:
; CHECK:   %lcmp.mod = icmp ne i32 %xtraiter, 0, !dbg ![[LOOP_LOC]]
; CHECK:   br i1 %lcmp.mod, label %for.body.i.epil.preheader, label %lee1.exit.loopexit, !dbg ![[LOOP_LOC]]
; CHECK: for.body.i.epil.preheader:
; CHECK:   br label %for.body.i.epil, !dbg ![[LOOP_LOC]]
; CHECK: for.body.i.epil:
;; Ensure that when we clone the div/add/add and its following dbg.values,
;; those dbg.values are remapped to the duplicated adds, not the originals.
; CHECK:      %div.i.epil = sdiv i32 %t.08.i.epil, 2,
; CHECK-NEXT: %add.i.epil = add i32 %t.08.i.epil, %a,
; CHECK-NEXT: %add1.i.epil = add i32 %add.i.epil, %div.i.epil,
; CHECK-NEXT: #dbg_value(i32 %add1.i.epil,
; CHECK-NEXT: %inc.i.epil = add nuw i32 %i.09.i.epil, 1, !dbg !36
; CHECK-NEXT: #dbg_value(i32 %inc.i.epil,
; CHECK-NEXT: #dbg_value(i32 %inc.i.epil,
; CHECK-NEXT: #dbg_value(i32 %add1.i.epil,

; CHECK: lee1.exit.loopexit:
; CHECK:   br label %lee1.exit, !dbg ![[EXIT_LOC:[0-9]+]]

; CHECK-DAG: ![[LOOP_LOC]] = !DILocation(line: 5, column: 3, scope: !{{.*}}, inlinedAt: !{{.*}})
; CHECK-DAG: ![[EXIT_LOC]] = !DILocation(line: 11, column: 12, scope: !{{.*}}, inlinedAt: !{{.*}})

; Function Attrs: nounwind readnone
define i32 @goo(i32 %a, i32 %b) local_unnamed_addr #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !13, metadata !15), !dbg !16
  tail call void @llvm.dbg.value(metadata i32 %b, i64 0, metadata !14, metadata !15), !dbg !17
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !18, metadata !15), !dbg !26
  tail call void @llvm.dbg.value(metadata i32 %b, i64 0, metadata !21, metadata !15), !dbg !28
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !22, metadata !15), !dbg !29
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !23, metadata !15), !dbg !30
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !23, metadata !15), !dbg !30
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !22, metadata !15), !dbg !29
  %cmp7.i = icmp eq i32 %b, 0, !dbg !31
  br i1 %cmp7.i, label %lee1.exit, label %for.body.i.preheader, !dbg !33

for.body.i.preheader:                             ; preds = %entry
  br label %for.body.i, !dbg !34

for.body.i:                                       ; preds = %for.body.i.preheader, %for.body.i
  %i.09.i = phi i32 [ %inc.i, %for.body.i ], [ 0, %for.body.i.preheader ]
  %t.08.i = phi i32 [ %add1.i, %for.body.i ], [ 0, %for.body.i.preheader ]
  %div.i = sdiv i32 %t.08.i, 2, !dbg !34
  %add.i = add i32 %t.08.i, %a, !dbg !35
  %add1.i = add i32 %add.i, %div.i, !dbg !36
  tail call void @llvm.dbg.value(metadata i32 %add1.i, i64 0, metadata !22, metadata !15), !dbg !29
  %inc.i = add nuw i32 %i.09.i, 1, !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %inc.i, i64 0, metadata !23, metadata !15), !dbg !30
  tail call void @llvm.dbg.value(metadata i32 %inc.i, i64 0, metadata !23, metadata !15), !dbg !30
  tail call void @llvm.dbg.value(metadata i32 %add1.i, i64 0, metadata !22, metadata !15), !dbg !29
  %exitcond.i = icmp eq i32 %inc.i, %b, !dbg !31
  br i1 %exitcond.i, label %lee1.exit.loopexit, label %for.body.i, !dbg !33, !llvm.loop !38

lee1.exit.loopexit:                               ; preds = %for.body.i
  %add1.i.lcssa = phi i32 [ %add1.i, %for.body.i ]
  br label %lee1.exit, !dbg !41

lee1.exit:                                        ; preds = %lee1.exit.loopexit, %entry
  %t.0.lcssa.i = phi i32 [ 0, %entry ], [ %add1.i.lcssa, %lee1.exit.loopexit ]
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !44, metadata !15), !dbg !47
  tail call void @llvm.dbg.value(metadata i32 %b, i64 0, metadata !45, metadata !15), !dbg !48
  %add.i4 = add nsw i32 %b, %a, !dbg !41
  %sub.i = sub nsw i32 %a, %b, !dbg !49
  %mul.i = mul nsw i32 %add.i4, %sub.i, !dbg !50
  %add = add nsw i32 %t.0.lcssa.i, %mul.i, !dbg !51
  ret i32 %add, !dbg !52
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+neon,+strict-align,+vfp3,-crypto,-fp-armv8,-fp16,-vfp4" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Snapdragon LLVM ARM Compiler 4.0.5 (based on llvm.org 4.0+)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/prj/llvm-arm/scratch1/zhaoshiz/bugs/debug-symbol")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"Snapdragon LLVM ARM Compiler 4.0.5 (based on llvm.org 4.0+)"}
!8 = distinct !DISubprogram(name: "goo", scope: !1, file: !1, line: 23, type: !9, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 23, type: !11)
!14 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !1, line: 23, type: !11)
!15 = !DIExpression()
!16 = !DILocation(line: 23, column: 14, scope: !8)
!17 = !DILocation(line: 23, column: 21, scope: !8)
!18 = !DILocalVariable(name: "a", arg: 1, scope: !19, file: !1, line: 3, type: !11)
!19 = distinct !DISubprogram(name: "lee1", scope: !1, file: !1, line: 3, type: !9, isLocal: true, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !20)
!20 = !{!18, !21, !22, !23}
!21 = !DILocalVariable(name: "b", arg: 2, scope: !19, file: !1, line: 3, type: !11)
!22 = !DILocalVariable(name: "t", scope: !19, file: !1, line: 4, type: !11)
!23 = !DILocalVariable(name: "i", scope: !24, file: !1, line: 5, type: !25)
!24 = distinct !DILexicalBlock(scope: !19, file: !1, line: 5, column: 3)
!25 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!26 = !DILocation(line: 3, column: 22, scope: !19, inlinedAt: !27)
!27 = distinct !DILocation(line: 24, column: 27, scope: !8)
!28 = !DILocation(line: 3, column: 29, scope: !19, inlinedAt: !27)
!29 = !DILocation(line: 4, column: 7, scope: !19, inlinedAt: !27)
!30 = !DILocation(line: 5, column: 17, scope: !24, inlinedAt: !27)
!31 = !DILocation(line: 5, column: 23, scope: !32, inlinedAt: !27)
!32 = distinct !DILexicalBlock(scope: !24, file: !1, line: 5, column: 3)
!33 = !DILocation(line: 5, column: 3, scope: !24, inlinedAt: !27)
!34 = !DILocation(line: 6, column: 13, scope: !32, inlinedAt: !27)
!35 = !DILocation(line: 6, column: 11, scope: !32, inlinedAt: !27)
!36 = !DILocation(line: 6, column: 7, scope: !32, inlinedAt: !27)
!37 = !DILocation(line: 5, column: 28, scope: !32, inlinedAt: !27)
!38 = distinct !{!38, !39, !40}
!39 = !DILocation(line: 5, column: 3, scope: !24, inlinedAt: !27)
!40 = !DILocation(line: 6, column: 14, scope: !24, inlinedAt: !27)
!41 = !DILocation(line: 11, column: 12, scope: !42, inlinedAt: !46)
!42 = distinct !DISubprogram(name: "lee2", scope: !1, file: !1, line: 10, type: !9, isLocal: true, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !43)
!43 = !{!44, !45}
!44 = !DILocalVariable(name: "a", arg: 1, scope: !42, file: !1, line: 10, type: !11)
!45 = !DILocalVariable(name: "b", arg: 2, scope: !42, file: !1, line: 10, type: !11)
!46 = distinct !DILocation(line: 24, column: 40, scope: !8)
!47 = !DILocation(line: 10, column: 22, scope: !42, inlinedAt: !46)
!48 = !DILocation(line: 10, column: 29, scope: !42, inlinedAt: !46)
!49 = !DILocation(line: 11, column: 20, scope: !42, inlinedAt: !46)
!50 = !DILocation(line: 11, column: 16, scope: !42, inlinedAt: !46)
!51 = !DILocation(line: 24, column: 38, scope: !8)
!52 = !DILocation(line: 24, column: 3, scope: !8)
