; RUN: opt %s -S -o - -passes=loop-unroll | FileCheck %s
; RUN: opt %s -S -o - -passes=loop-unroll --try-experimental-debuginfo-iterators | FileCheck %s
; generated at -O3 from:
; void f() {
;   for (int i = 1; i <=32; i <<=2 )
;     bar(i>>1);
; }
source_filename = "/tmp/loop.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

; Function Attrs: nounwind ssp uwtable
define void @f() local_unnamed_addr #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !15
  br label %for.body, !dbg !16

for.cond.cleanup:                                 ; preds = %for.body
  ret void, !dbg !17

for.body:                                         ; preds = %entry, %for.body
  %i.04 = phi i32 [ 1, %entry ], [ %shl, %for.body ]
  tail call void @llvm.dbg.value(metadata i32 %i.04, metadata !12, metadata !DIExpression()), !dbg !15
  %shr = ashr i32 %i.04, 1, !dbg !18

  ; The loop gets unrolled entirely.
  ; CHECK: #dbg_value(i32 1, !12, !DIExpression(), !15
  ; CHECK: #dbg_value(i32 4, !12, !DIExpression(), !15
  ; CHECK: #dbg_value(i32 16, !12, !DIExpression(), !15
  ; CHECK: #dbg_value(i32 64, !12, !DIExpression(), !15

  %call = tail call i32 (i32, ...) @bar(i32 %shr) #3, !dbg !20
  %shl = shl i32 %i.04, 2, !dbg !21
  tail call void @llvm.dbg.value(metadata i32 %shl, metadata !12, metadata !DIExpression()), !dbg !15
  %cmp = icmp slt i32 %shl, 33, !dbg !22
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !16, !llvm.loop !23
}

declare i32 @bar(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 317113) (llvm/trunk 317122)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/loop.c", directory: "/Data/llvm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 (trunk 317113) (llvm/trunk 317122)"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !11)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{!12}
!12 = !DILocalVariable(name: "i", scope: !13, file: !1, line: 2, type: !14)
!13 = distinct !DILexicalBlock(scope: !8, file: !1, line: 2, column: 3)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 2, column: 12, scope: !13)
!16 = !DILocation(line: 2, column: 3, scope: !13)
!17 = !DILocation(line: 4, column: 1, scope: !8)
!18 = !DILocation(line: 3, column: 10, scope: !19)
!19 = distinct !DILexicalBlock(scope: !13, file: !1, line: 2, column: 3)
!20 = !DILocation(line: 3, column: 5, scope: !19)
!21 = !DILocation(line: 2, column: 29, scope: !19)
!22 = !DILocation(line: 2, column: 21, scope: !19)
!23 = distinct !{!23, !16, !24}
!24 = !DILocation(line: 3, column: 13, scope: !13)
