; RUN: opt -S -loop-reduce %s | FileCheck %s

;; Ensure that scev-based salvaging in LSR does not select an IV containing
;; an 'undef' element.

target triple = "x86_64-unknown-linux-gnu"

define i16 @n(i1 %c1) !dbg !8 {
entry:
  br i1 %c1, label %m, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i16 [ %ivdec, %for.body ], [ 14, %entry ]
  %ivdec = sub i16 %iv, 1
  call void @llvm.dbg.value(metadata i16 %iv, metadata !21, metadata !DIExpression()), !dbg !19
  br label %for.body

m:                                                ; preds = %m, %entry
  %0 = phi i16 [ 3, %m ], [ 6, %entry ]
  %gg = add i16 %0, 23
  ; CHECK: call void @llvm.dbg.value(metadata i16 poison, metadata !{{[0-9]+}}, metadata !DIExpression()),
  call void @llvm.dbg.value(metadata i16 %0, metadata !14, metadata !DIExpression()), !dbg !19
  br label %m
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduced.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 14.0.0"}
!8 = distinct !DISubprogram(name: "n", scope: !1, file: !1, line: 18, type: !9, scopeLine: 18, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILabel(scope: !8, name: "m", file: !1, line: 22)
!14 = !DILocalVariable(name: "k", arg: 2, scope: !15, file: !1, line: 9, type: !11)
!15 = distinct !DISubprogram(name: "i", scope: !1, file: !1, line: 9, type: !16, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!11, !11, !11}
!18 = !{!14}
!19 = !DILocation(line: 0, scope: !15, inlinedAt: !20)
!20 = distinct !DILocation(line: 23, scope: !8)
!21 = !DILocalVariable(name: "x", arg: 2, scope: !15, file: !1, line: 1, type: !11)
