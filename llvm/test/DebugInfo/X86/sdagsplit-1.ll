; RUN: llc %s -stop-after=livedebugvars -o %t -experimental-debug-variable-locations=true
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
; CHECK-DAG: DBG_VALUE_LIST !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 0, 32), ${{[a-z]+}}, debug-location
; CHECK-DAG: DBG_VALUE_LIST !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 32, 32), ${{[a-z]+}}, debug-location

; ModuleID = 'sdagsplit-1.c'
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386"

; Function Attrs: nounwind readnone
define i64 @foo(i64 %a, i64 %b) local_unnamed_addr #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i64 %a, metadata !13, metadata !16), !dbg !17
  tail call void @llvm.dbg.value(metadata i64 %b, metadata !14, metadata !16), !dbg !18
  %cmp = icmp eq i64 %a, %b, !dbg !20
  %add = add nsw i64 %b, 1, !dbg !22
  tail call void @llvm.dbg.value(metadata i64 %add, metadata !15, metadata !16), !dbg !19
  %retval.0 = select i1 %cmp, i64 %add, i64 0, !dbg !23
  ret i64 %retval.0, !dbg !24
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "sdagsplit-1.c", directory: "/X86")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !11}
!11 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!14 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !1, line: 2, type: !11)
!15 = !DILocalVariable(name: "res", scope: !8, file: !1, line: 4, type: !11)
!16 = !DIExpression()
!17 = !DILocation(line: 2, column: 26, scope: !8)
!18 = !DILocation(line: 2, column: 39, scope: !8)
!19 = !DILocation(line: 4, column: 12, scope: !8)
!20 = !DILocation(line: 5, column: 9, scope: !21)
!21 = distinct !DILexicalBlock(scope: !8, file: !1, line: 5, column: 7)
!22 = !DILocation(line: 4, column: 19, scope: !8)
!23 = !DILocation(line: 5, column: 7, scope: !8)
!24 = !DILocation(line: 8, column: 1, scope: !8)
