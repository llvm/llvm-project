; RUN: llc < %s -prefetch-hints-file=%S/insert-prefetch.afdo | FileCheck %s
; RUN: llc < %s -prefetch-hints-file=%S/insert-prefetch-other.afdo | FileCheck %s -check-prefix=OTHERS
;
; original source, compiled with -O3 -gmlt -fdebug-info-for-profiling:
; int sum(int* arr, int pos1, int pos2) {
;   return arr[pos1] + arr[pos2];
; }
;
; NOTE: debug line numbers were adjusted such that the function would start
; at line 15 (an arbitrary number). The sample profile file format uses
; offsets from the start of the symbol instead of file-relative line numbers.
; The .afdo file reflects that - the instructions are offset '1'.
;
; ModuleID = 'test.cc'
source_filename = "test.cc"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @sum(i32* %arr, i32 %pos1, i32 %pos2) !dbg !35 !prof !37 {
entry:
  %idxprom = sext i32 %pos1 to i64, !dbg !38
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom, !dbg !38
  %0 = load i32, i32* %arrayidx, align 4, !dbg !38, !tbaa !39
  %idxprom1 = sext i32 %pos2 to i64, !dbg !43
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i64 %idxprom1, !dbg !43
  %1 = load i32, i32* %arrayidx2, align 4, !dbg !43, !tbaa !39
  %add = add nsw i32 %1, %0, !dbg !44
  ret i32 %add, !dbg !45
}

attributes #0 = { "target-cpu"="x86-64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, debugInfoForProfiling: true)
!1 = !DIFile(filename: "test.cc", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"ProfileSummary", !7}
!7 = !{!8, !9, !10, !11, !12, !13, !14, !15}
!8 = !{!"ProfileFormat", !"SampleProfile"}
!9 = !{!"TotalCount", i64 0}
!10 = !{!"MaxCount", i64 0}
!11 = !{!"MaxInternalCount", i64 0}
!12 = !{!"MaxFunctionCount", i64 0}
!13 = !{!"NumCounts", i64 2}
!14 = !{!"NumFunctions", i64 1}
!15 = !{!"DetailedSummary", !16}
!16 = !{!17, !18, !19, !20, !21, !22, !22, !23, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32}
!17 = !{i32 10000, i64 0, i32 0}
!18 = !{i32 100000, i64 0, i32 0}
!19 = !{i32 200000, i64 0, i32 0}
!20 = !{i32 300000, i64 0, i32 0}
!21 = !{i32 400000, i64 0, i32 0}
!22 = !{i32 500000, i64 0, i32 0}
!23 = !{i32 600000, i64 0, i32 0}
!24 = !{i32 700000, i64 0, i32 0}
!25 = !{i32 800000, i64 0, i32 0}
!26 = !{i32 900000, i64 0, i32 0}
!27 = !{i32 950000, i64 0, i32 0}
!28 = !{i32 990000, i64 0, i32 0}
!29 = !{i32 999000, i64 0, i32 0}
!30 = !{i32 999900, i64 0, i32 0}
!31 = !{i32 999990, i64 0, i32 0}
!32 = !{i32 999999, i64 0, i32 0}
!33 = !{!"clang version 7.0.0 (trunk 322593) (llvm/trunk 322526)"}
!35 = distinct !DISubprogram(name: "sum", linkageName: "sum", scope: !1, file: !1, line: 15, type: !36, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!36 = !DISubroutineType(types: !2)
!37 = !{!"function_entry_count", i64 -1}
!38 = !DILocation(line: 16, column: 10, scope: !35)
!39 = !{!40, !40, i64 0}
!40 = !{!"int", !41, i64 0}
!41 = !{!"omnipotent char", !42, i64 0}
!42 = !{!"Simple C++ TBAA"}
!43 = !DILocation(line: 16, column: 22, scope: !35)
!44 = !DILocation(line: 16, column: 20, scope: !35)
!45 = !DILocation(line: 16, column: 3, scope: !35)

;CHECK-LABEL: sum:
;CHECK:       # %bb.0:
;CHECK:       prefetchnta 42(%rdi,%rax,4)
;CHECK-NEXT:  prefetchnta (%rdi,%rax,4)
;CHECK-NEXT:  movl (%rdi,%rax,4), %eax
;CHECK-NEXT:  .loc 1 16 20 discriminator 2  # test.cc:16:20
;CHECK-NEXT:  prefetchnta -1(%rdi,%rcx,4)
;CHECK-NEXT:  addl (%rdi,%rcx,4), %eax
;CHECK-NEXT:  .loc 1 16 3                   # test.cc:16:3

;OTHERS-LABEL: sum:
;OTHERS:       # %bb.0:
;OTHERS:       prefetcht2 42(%rdi,%rax,4)
;OTHERS-NEXT:  prefetcht0 (%rdi,%rax,4)
;OTHERS-NEXT:  movl (%rdi,%rax,4), %eax
;OTHERS-NEXT:  .loc 1 16 20 discriminator 2  # test.cc:16:20
;OTHERS-NEXT:  prefetcht1 -1(%rdi,%rcx,4)
;OTHERS-NEXT:  addl (%rdi,%rcx,4), %eax
;OTHERS-NEXT:  .loc 1 16 3                   # test.cc:16:3
