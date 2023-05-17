; Tests that hot callsite threshold is set to 0 artifically for thinlto-prelink pipeline.
;
; Function `sum` is annotated with inline cost -1 and function `sum1` is
; annotated with inline cost 0, by function attribute `function-inline-cost`.
;
; `if.then` basic block is hot so the callsite threshold is set to 0.
; `while.body.split` basic block is cold so threshold is calculated using
; the rest of default heuristics, which should be sufficient to inline a
; function with 0 cost.

; RUN: opt < %s -pass-remarks=inline -pass-remarks-missed=inline -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/new-pm-thinlto-prelink-samplepgo-inline-threshold.prof -S | FileCheck %s

; RUN: opt < %s -pass-remarks=inline -pass-remarks-missed=inline -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/new-pm-thinlto-prelink-samplepgo-inline-threshold.prof -S 2>&1 | FileCheck %s -check-prefix=REMARK

; Original C++ test case
;
; #include <stdio.h>

; int sum(int x, int y) { return x + y; }

; int sum1(int x, int y) { return x + y; }

; int main() {
;   int s, i = 0;
;   while (i++ < 20000 * 20000) {
;     if (i != 100) s = sum(i, s);
;     s = sum1(i, s);
;   }
;   printf("sum is %d\n", s);
;   return 0;
; }
;

; REMARK: test.cc:14:9: '_Z4sum1ii' inlined into 'main' with (cost=0, threshold=45) at callsite main:4:9;
; REMARK: test.cc:13:23: '_Z3sumii' inlined into 'main' with (cost=-1, threshold=0) at callsite main:3:23.2;
; REMARK: test.cc:14:9: '_Z4sum1ii' not inlined into 'main' because too costly to inline (cost=0, threshold=0)

; ModuleID = 'test.cc'
source_filename = "test.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = constant [11 x i8] c"sum is %d\0A\00", align 1

define i32 @_Z3sumii(i32 %x, i32 %y) "function-inline-cost"="-1" #0 !dbg !7 {
entry:
  %add = add nsw i32 %y, %x, !dbg !10
  ret i32 %add, !dbg !11
}

define i32 @_Z4sum1ii(i32 %x, i32 %y) "function-inline-cost"="0" #0 !dbg !12 {
entry:
  %add = add nsw i32 %y, %x, !dbg !13
  ret i32 %add, !dbg !14
}

define i32 @main() #0 !dbg !15 {
entry:
  br label %while.body, !dbg !16

while.body:                                       ; preds = %entry, %if.end
  %inc14 = phi i32 [ 1, %entry ], [ %inc, %if.end ]
  %s.013 = phi i32 [ undef, %entry ], [ %phi.call, %if.end ]
  %cmp1.not = icmp eq i32 %inc14, 100, !dbg !17
  br i1 %cmp1.not, label %while.body.split, label %if.then, !dbg !18

while.body.split:                                 ; preds = %while.body
; CHECK-NOT: call noundef i32 @_Z4sum1ii
  %call211 = call noundef i32 @_Z4sum1ii(i32 100, i32 %s.013), !dbg !19
  br label %if.end, !dbg !18

if.then:                                          ; preds = %while.body
; CHECK-NOT: call i32 @_Z3sumii
; CHECK: call i32 @_Z4sum1ii
  %call = call i32 @_Z3sumii(i32 %inc14, i32 %s.013), !dbg !20
  %call212 = call i32 @_Z4sum1ii(i32 %inc14, i32 %call), !dbg !19
  br label %if.end, !dbg !21

if.end:                                           ; preds = %while.body.split, %if.then
  %phi.call = phi i32 [ %call211, %while.body.split ], [ %call212, %if.then ], !dbg !19
  %inc = add i32 %inc14, 1, !dbg !22
  %exitcond.not = icmp eq i32 %inc, 400000001, !dbg !23
  br i1 %exitcond.not, label %while.end, label %while.body, !dbg !16, !llvm.loop !24

while.end:                                        ; preds = %if.end
  %call3 = tail call i32 (ptr, ...) @printf(ptr dereferenceable(1) @.str, i32 %phi.call), !dbg !27
  ret i32 0, !dbg !28
}

declare i32 @printf(ptr nocapture noundef readonly, ...)

attributes #0 = {"use-sample-profile"}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 329fda39c507e8740978d10458451dcdb21563be)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cc", directory: "", checksumkind: CSK_MD5, checksum: "e8c44edbdcc2c41f9f891ac2b2ddd591")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "sum", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 6, column: 60, scope: !7)
!11 = !DILocation(line: 6, column: 51, scope: !7)
!12 = distinct !DISubprogram(name: "sum1", scope: !1, file: !1, line: 8, type: !8, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!13 = !DILocation(line: 8, column: 61, scope: !12)
!14 = !DILocation(line: 8, column: 52, scope: !12)
!15 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!16 = !DILocation(line: 12, column: 3, scope: !15)
!17 = !DILocation(line: 13, column: 11, scope: !15)
!18 = !DILocation(line: 13, column: 9, scope: !15)
!19 = !DILocation(line: 14, column: 9, scope: !15)
!20 = !DILocation(line: 13, column: 23, scope: !15)
!21 = !DILocation(line: 13, column: 19, scope: !15)
!22 = !DILocation(line: 12, column: 11, scope: !15)
!23 = !DILocation(line: 12, column: 14, scope: !15)
!24 = distinct !{!24, !16, !25, !26}
!25 = !DILocation(line: 15, column: 3, scope: !15)
!26 = !{!"llvm.loop.mustprogress"}
!27 = !DILocation(line: 16, column: 3, scope: !15)
!28 = !DILocation(line: 17, column: 3, scope: !15)
