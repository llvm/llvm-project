; Checks if indirect calls to static target functions that are actually
; dead in the new binary target (due to a profile collected from a slightly
; different binary) are properly traversed during ThinLTO liveness analysis.
; If the liveness analysis is changed to ignore indirect edges and the
; importer is changed to check liveness before importing, this test will
; need adjustment (in that case _ZL3foov should not be imported/promoted,
; and _ZL3barv can be internalized/removed).

; REQUIRES: x86-registered-target

; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/thinlto_samplepgo_icp2a.ll -o %t2a.bc
; RUN: opt -module-summary %p/Inputs/thinlto_samplepgo_icp2b.ll -o %t2b.bc

; Use -import-instr-limit=5 so that we don't import _ZL3barv, which would
; hide the problem.
; RUN: llvm-lto2 run -save-temps -import-instr-limit=5 -o %t3 %t.bc %t2a.bc %t2b.bc -r %t.bc,fptr,plx -r %t.bc,main,plx -r %t2a.bc,_ZL3barv,l -r %t2b.bc,_ZL3barv,pl -print-imports 2>&1 | FileCheck %s --check-prefix=IMPORTS2
; IMPORTS2-NOT: Import _ZL3barv
; IMPORTS2: Import _ZL3foov.llvm.0
; IMPORTS2-NOT: Import _ZL3barv
; RUN: llvm-nm %t3.2 | FileCheck %s --check-prefix=NM
; NM: _ZL3barv
; RUN: llvm-dis < %t3.2.2.internalize.bc | FileCheck %s --check-prefix=INTERNALIZE
; INTERNALIZE: define dso_local void @_ZL3barv

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fptr = local_unnamed_addr global void ()* null, align 8

; Function Attrs: norecurse uwtable
define i32 @main() local_unnamed_addr #0 !prof !34 {
entry:
  %0 = load void ()*, void ()** @fptr, align 8
; ICALL-PROM:   br i1 %{{[0-9]+}}, label %if.true.direct_targ, label %if.false.orig_indirect
  tail call void %0(), !prof !40
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3,!4}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 297016)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "main.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"ProfileSummary", !5}
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13}
!6 = !{!"ProfileFormat", !"SampleProfile"}
!7 = !{!"TotalCount", i64 3003}
!8 = !{!"MaxCount", i64 3000}
!9 = !{!"MaxInternalCount", i64 0}
!10 = !{!"MaxFunctionCount", i64 0}
!11 = !{!"NumCounts", i64 3}
!12 = !{!"NumFunctions", i64 1}
!13 = !{!"DetailedSummary", !14}
!14 = !{!15, !16, !17, !18, !19, !20, !20, !21, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30}
!15 = !{i32 10000, i64 3000, i32 1}
!16 = !{i32 100000, i64 3000, i32 1}
!17 = !{i32 200000, i64 3000, i32 1}
!18 = !{i32 300000, i64 3000, i32 1}
!19 = !{i32 400000, i64 3000, i32 1}
!20 = !{i32 500000, i64 3000, i32 1}
!21 = !{i32 600000, i64 3000, i32 1}
!22 = !{i32 700000, i64 3000, i32 1}
!23 = !{i32 800000, i64 3000, i32 1}
!24 = !{i32 900000, i64 3000, i32 1}
!25 = !{i32 950000, i64 3000, i32 1}
!26 = !{i32 990000, i64 3000, i32 1}
!27 = !{i32 999000, i64 3000, i32 1}
!28 = !{i32 999900, i64 2, i32 2}
!29 = !{i32 999990, i64 2, i32 2}
!30 = !{i32 999999, i64 2, i32 2}
!31 = !{!"clang version 5.0.0 (trunk 297016)"}
!34 = !{!"function_entry_count", i64 1}
!40 = !{!"VP", i32 0, i64 3000, i64 -8789629626369651636, i64 3000}
