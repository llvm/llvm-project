; RUN: llc %s -stop-after=livedebugvars -o %t
; RUN: cat %t | FileCheck %s
;
; Test that we can emit debug info for large values that are split
; up across multiple registers by the SelectionDAG type legalizer.
;
;    // Compile with -O1 -m32.
;    long long foo (long long a, long long b)
;    {
;      long long res = b+1;
;      if ( a == b )
;        return res;
;      return 0;
;    }
;
; CHECK-DAG: DBG_VALUE debug-use $r{{[0-9]+}}, debug-use $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 32, 16), debug-location !{{[0-9]+}}
; CHECK-DAG: DBG_VALUE debug-use $r{{[0-9]+}}, debug-use $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 48, 16), debug-location !{{[0-9]+}}
; CHECK-DAG: DBG_VALUE debug-use $r{{[0-9]+}}, debug-use $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 0, 16), debug-location !{{[0-9]+}}
; CHECK-DAG: DBG_VALUE debug-use $r{{[0-9]+}}, debug-use $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 16, 16), debug-location !{{[0-9]+}}

; ModuleID = 'sdagsplit-1.c'
target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430"

; Function Attrs: nounwind readnone
define i64 @foo(i64 %a, i64 %b) local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata i64 %a, metadata !12, metadata !15), !dbg !16
  tail call void @llvm.dbg.value(metadata i64 %b, metadata !13, metadata !15), !dbg !17
  tail call void @llvm.dbg.value(metadata i64 %add, metadata !14, metadata !15), !dbg !18
  %cmp = icmp eq i64 %a, %b, !dbg !19
  %add = add nsw i64 %b, 1, !dbg !21
  %retval.0 = select i1 %cmp, i64 %add, i64 0, !dbg !22
  ret i64 %retval.0, !dbg !23
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "sdagsplit-1.c", directory: "/MSP430")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{!"clang version 6.0.0 "}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10}
!10 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !DILocalVariable(name: "b", arg: 2, scope: !7, file: !1, line: 2, type: !10)
!14 = !DILocalVariable(name: "res", scope: !7, file: !1, line: 4, type: !10)
!15 = !DIExpression()
!16 = !DILocation(line: 2, column: 26, scope: !7)
!17 = !DILocation(line: 2, column: 39, scope: !7)
!18 = !DILocation(line: 4, column: 12, scope: !7)
!19 = !DILocation(line: 5, column: 9, scope: !20)
!20 = distinct !DILexicalBlock(scope: !7, file: !1, line: 5, column: 7)
!21 = !DILocation(line: 4, column: 19, scope: !7)
!22 = !DILocation(line: 5, column: 7, scope: !7)
!23 = !DILocation(line: 8, column: 1, scope: !7)
