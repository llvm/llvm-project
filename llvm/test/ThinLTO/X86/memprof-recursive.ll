;; Test recursion handling during cloning.
;;
;; See llvm/test/Transforms/MemProfContextDisambiguation/recursive.ll for
;; information on how the test was created.

; RUN: opt -thinlto-bc %s >%t.o

;; Check behavior when we enable cloning of contexts involved with recursive
;; cycles, but not through the cycle itself. I.e. until full support for
;; recursion is added, the cloned recursive call from C back to B (line 12) will
;; not be updated to call a clone.
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,_Z1Dv,plx \
; RUN:  -r=%t.o,_Z1Ci,plx \
; RUN:  -r=%t.o,_Z1Bi,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:	-memprof-allow-recursive-callsites=true \
; RUN:  -o %t.out 2>&1 | FileCheck %s \
; RUN:  --implicit-check-not "memprof_recursive3.cc:12:10: call in clone _Z1Ci.memprof.1 assigned" \
; RUN:  --check-prefix=ALLOW-RECUR-CALLSITES --check-prefix=ALLOW-RECUR-CONTEXTS

;; Skipping recursive callsites should result in no cloning.
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,_Z1Dv,plx \
; RUN:  -r=%t.o,_Z1Ci,plx \
; RUN:  -r=%t.o,_Z1Bi,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:	-memprof-allow-recursive-callsites=false \
; RUN:  -o %t.out 2>&1 | FileCheck %s --allow-empty \
; RUN:  --implicit-check-not "memprof_recursive3.cc:12:10: call in clone _Z1Ci.memprof.1 assigned" \
; RUN:  --implicit-check-not="created clone" \
; RUN:	--implicit-check-not="marked with memprof allocation attribute cold"

;; Check the default behavior (disabled recursive callsites).
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,_Z1Dv,plx \
; RUN:  -r=%t.o,_Z1Ci,plx \
; RUN:  -r=%t.o,_Z1Bi,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -o %t.out 2>&1 | FileCheck %s --allow-empty \
; RUN:  --implicit-check-not "memprof_recursive3.cc:12:10: call in clone _Z1Ci.memprof.1 assigned" \
; RUN:  --implicit-check-not="created clone" \
; RUN:	--implicit-check-not="marked with memprof allocation attribute cold"

;; Skipping recursive contexts should prevent spurious call to cloned version of
;; B from the context starting at memprof_recursive.cc:19:13, which is actually
;; recursive (until that support is added).
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,_Z1Dv,plx \
; RUN:  -r=%t.o,_Z1Ci,plx \
; RUN:  -r=%t.o,_Z1Bi,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:	-memprof-allow-recursive-callsites=true \
; RUN:	-memprof-allow-recursive-contexts=false \
; RUN:  -o %t.out 2>&1 | FileCheck %s \
; RUN:  --implicit-check-not "memprof_recursive3.cc:12:10: call in clone _Z1Ci.memprof.1 assigned" \
; RUN:  --check-prefix=ALLOW-RECUR-CALLSITES --check-prefix=SKIP-RECUR-CONTEXTS

; ALLOW-RECUR-CALLSITES: memprof_recursive.cc:4:0: created clone _Z1Dv.memprof.1
; ALLOW-RECUR-CALLSITES: memprof_recursive.cc:5:10: call in clone _Z1Dv marked with memprof allocation attribute notcold
; ALLOW-RECUR-CALLSITES: memprof_recursive.cc:5:10: call in clone _Z1Dv.memprof.1 marked with memprof allocation attribute cold
; ALLOW-RECUR-CALLSITES: memprof_recursive.cc:8:0: created clone _Z1Ci.memprof.1
; ALLOW-RECUR-CALLSITES: memprof_recursive.cc:10:12: call in clone _Z1Ci.memprof.1 assigned to call function clone _Z1Dv.memprof.1
; ALLOW-RECUR-CALLSITES: memprof_recursive.cc:14:0: created clone _Z1Bi.memprof.1
; ALLOW-RECUR-CALLSITES: memprof_recursive.cc:15:10: call in clone _Z1Bi.memprof.1 assigned to call function clone _Z1Ci.memprof.1
;; We should only call the cold clone for the recursive context if we enabled
;; recursive contexts via -memprof-allow-recursive-contexts=true (default).
; ALLOW-RECUR-CONTEXTS: memprof_recursive.cc:19:13: call in clone main assigned to call function clone _Z1Bi.memprof.1
; SKIP-RECUR-CONTEXTS-NOT: memprof_recursive.cc:19:13: call in clone main assigned to call function clone _Z1Bi.memprof.1
; ALLOW-RECUR-CALLSITES: memprof_recursive.cc:20:13: call in clone main assigned to call function clone _Z1Bi.memprof.1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @_Z1Dv() !dbg !3 {
entry:
  %call = tail call ptr @_Znam(i64 10), !dbg !6, !memprof !7, !callsite !14
  ret ptr null
}

define ptr @_Z1Ci(i32 %n) !dbg !15 {
entry:
  %call = tail call ptr @_Z1Dv(), !dbg !16, !callsite !17
  br label %return

if.end:                                           ; No predecessors!
  %call1 = tail call ptr @_Z1Bi(i32 0), !dbg !18, !callsite !19
  br label %return

return:                                           ; preds = %if.end, %entry
  ret ptr null
}

define ptr @_Z1Bi(i32 %n) !dbg !20 {
entry:
  %call = tail call ptr @_Z1Ci(i32 0), !dbg !21, !callsite !22
  ret ptr null
}

define i32 @main() {
entry:
  %call = tail call ptr @_Z1Bi(i32 0), !dbg !23, !callsite !25
  %call1 = tail call ptr @_Z1Bi(i32 0), !dbg !26, !callsite !27
  %call2 = tail call ptr @_Z1Bi(i32 0), !dbg !28, !callsite !29
  ret i32 0
}

declare ptr @_Znam(i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0git (https://github.com/llvm/llvm-project.git 7aec6dc477f8148ed066d10dfc7a012a51b6599c)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof_recursive.cc", directory: ".", checksumkind: CSK_MD5, checksum: "2f15f63b187a0e0d40e7fdd18b10576a")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "D", linkageName: "_Z1Dv", scope: !1, file: !1, line: 4, type: !4, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 5, column: 10, scope: !3)
!7 = !{!8, !10, !12}
!8 = !{!9, !"cold"}
!9 = !{i64 6541423618768552252, i64 -200552803509692312, i64 -2954124005641725917, i64 6307901912192269588}
!10 = !{!11, !"notcold"}
!11 = !{i64 6541423618768552252, i64 -200552803509692312, i64 -2954124005641725917, i64 -7155190423157709404, i64 -2954124005641725917, i64 8632435727821051414}
!12 = !{!13, !"cold"}
!13 = !{i64 6541423618768552252, i64 -200552803509692312, i64 -2954124005641725917, i64 -7155190423157709404, i64 -2954124005641725917, i64 -3421689549917153178}
!14 = !{i64 6541423618768552252}
!15 = distinct !DISubprogram(name: "C", linkageName: "_Z1Ci", scope: !1, file: !1, line: 8, type: !4, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!16 = !DILocation(line: 10, column: 12, scope: !15)
!17 = !{i64 -200552803509692312}
!18 = !DILocation(line: 12, column: 10, scope: !15)
!19 = !{i64 -7155190423157709404}
!20 = distinct !DISubprogram(name: "B", linkageName: "_Z1Bi", scope: !1, file: !1, line: 14, type: !4, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 15, column: 10, scope: !20)
!22 = !{i64 -2954124005641725917}
!23 = !DILocation(line: 18, column: 13, scope: !24)
!24 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 17, type: !4, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!25 = !{i64 8632435727821051414}
!26 = !DILocation(line: 19, column: 13, scope: !24)
!27 = !{i64 -3421689549917153178}
!28 = !DILocation(line: 20, column: 13, scope: !24)
!29 = !{i64 6307901912192269588}
