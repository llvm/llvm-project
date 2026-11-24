;; Test to make sure that missing frames in memprof profiles due to indirect tail
;; calls are identified and handled when the profiled target of the indirect
;; tail call has local linkage, and the VP metadata contains the GUID of the
;; target's original name without the source file prefix (as is the case when
;; the VP metadata comes from a SamplePGO profile). The thin link should
;; translate the original name GUID to the actual GUID of the local target, and
;; the ThinLTO backend should then be able to locate the synthesized callsite
;; record and the target function for ICP. The local target is in a different
;; module than the indirect tail call, so it is imported and promoted (renamed)
;; in the ThinLTO backend for the module containing the indirect tail call.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o
; RUN: opt -thinlto-bc %t/baz.ll -o %t/baz.o

;; The summary edge for the profiled target of the indirect tail call should be
;; marked as a tail call (as well as indirect), and refer to the original name
;; GUID of the local target (MD5("_ZL3barv")).
; RUN: llvm-dis %t/baz.o -o - | FileCheck %s --check-prefix=SUMMARY
; SUMMARY: gv: (name: "_Z3bazPFPvvE", {{.*}} calls: ((callee: ^[[ORIG:[0-9]+]], hotness: none, tail: 1, indirect: 1))
; SUMMARY: ^[[ORIG]] = gv: (guid: 13960636260896978002)

;; In-process ThinLTO.
; RUN: llvm-lto2 run %t/main.o %t/baz.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/baz.o,_Z3bazPFPvvE,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3bazPFPvvE, \
; RUN:  -r=%t/main.o,_Znam, \
; RUN:  -thinlto-threads=1 \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -stats -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefixes=STATS,STATS-INPROC,REMARKS \
; RUN:  --implicit-check-not="cannot promote"

; RUN: llvm-dis %t.out.2.4.opt.bc -o - | FileCheck %s --check-prefix=IR

;; Distributed ThinLTO.
; RUN: llvm-lto2 run %t/main.o %t/baz.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t/baz.o,_Z3bazPFPvvE,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3bazPFPvvE, \
; RUN:  -r=%t/main.o,_Znam, \
; RUN:  -stats \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=STATS

;; Run ThinLTO backend for baz.o. The distributed index does not contain the
;; original names of local functions, so the backend must compute the original
;; name GUID of the synthesized record's (promoted) callee to match it with the
;; profiled target.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -summary-file=%t/baz.o.thinlto.bc -memprof-import-summary=%t/baz.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -stats %t/baz.o -S -o %t.be.ll 2>&1 | FileCheck %s --check-prefixes=STATS-BE,REMARKS \
; RUN:  --implicit-check-not="cannot promote"
; RUN: FileCheck %s --check-prefix=IR < %t.be.ll

; REMARKS-DAG: call in clone _Z3bazPFPvvE promoted and assigned to call function clone _ZL3barv.llvm.{{[0-9]+}}{{$}}
; REMARKS-DAG: call in clone _Z3bazPFPvvE.memprof.1 promoted and assigned to call function clone _ZL3barv.llvm.{{[0-9]+}}.memprof.1

;; The missing frame is found once for each of the 2 profiled callers (main's
;; calls to baz).
; STATS: 2 memprof-context-disambiguation - Number of profiled callees found via tail calls
; STATS: 2 memprof-context-disambiguation - Number of tail calls in chains to profiled callees that are indirect (thin link)
;; In-process, baz is also imported into main's module, so its indirect tail
;; call is recorded for ICP in both backends.
; STATS-INPROC: 2 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend
; STATS-BE: 1 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

;; Both versions of baz compare against the original (promoted) bar.
; IR: define {{.*}} @_Z3bazPFPvvE(
; IR:   icmp eq ptr %fp, @[[BAR:_ZL3barv.llvm.[0-9]+]]
; IR: define {{.*}} @_Z3bazPFPvvE.memprof.1(
; IR:   icmp eq ptr %fp, @[[BAR]]{{$}}

;--- baz.ll
source_filename = "baz.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; The indirect tail call to bar (via the function pointer) is missing from the
;; profiled contexts, which have main calling bar directly.
define ptr @_Z3bazPFPvvE(ptr %fp) {
entry:
  %call = tail call ptr %fp(), !prof !0
  ret ptr %call
}

;; VP metadata with a single candidate, the original name GUID of the local
;; function _ZL3barv (i.e. MD5("_ZL3barv"), without the "main.cc;" prefix).
!0 = !{!"VP", i32 0, i64 4, i64 -4486107812812573614, i64 4}

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  ;; The first call to baz is part of a not cold context, and the second is
  ;; part of a cold context.
  %call = call ptr @_Z3bazPFPvvE(ptr @_ZL3barv), !callsite !6
  %call1 = call ptr @_Z3bazPFPvvE(ptr @_ZL3barv), !callsite !7
  ret i32 0
}

declare ptr @_Z3bazPFPvvE(ptr)

;; Local function only called indirectly.
define internal ptr @_ZL3barv() {
entry:
  %call = tail call ptr @_Znam(i64 10) #0, !memprof !1, !callsite !5
  ret ptr %call
}

declare ptr @_Znam(i64)

attributes #0 = { builtin allocsize(0) }

!1 = !{!2, !4}
!2 = !{!3, !"notcold"}
!3 = !{i64 3186456655321080972, i64 8632435727821051414}
!4 = !{!8, !"cold"}
!8 = !{i64 3186456655321080972, i64 -3421689549917153178}
!5 = !{i64 3186456655321080972}
!6 = !{i64 8632435727821051414}
!7 = !{i64 -3421689549917153178}
