;; Test that the thin link does not synthesize callsite records for tail call
;; chains that are discarded because the profiled callee is reachable through
;; multiple chains. Such records would not be extended when the function is
;; cloned, so they would have fewer clone entries than the function's other
;; records, and the ThinLTO backend could then fail to clone the function when
;; applying ICP for an indirect tail call whose profiled targets include the
;; stale record's callee (leaving undefined references to the function's clone
;; from its callers).
;;
;; @mid has two indirect tail calls. The first (VP target @t1) orders @t1 before
;; @t2 in @mid's summary call edges. @leafA is reachable from @mid via both @t1
;; and @t2, so its tail call chain search finds the chain through @t1 before
;; discovering the second chain through @t2 and giving up. Without the fix,
;; records for the tail calls in the first chain (@mid to @t1, and @t1 to
;; @leafA) were synthesized before the second chain was found and left in the
;; summary. @leafB is reachable only via @t2, so @mid is cloned for its
;; contexts with a record for @t2. The second indirect tail call's VP lists @t2
;; before @t1, so the backend would see a stale @t1 record (with a single clone
;; entry) last.
;;
;; Note that the test is intentionally order-sensitive: it relies on @t1
;; preceding @t2 in @mid's summary call edges (so the ambiguous chain's record
;; is synthesized first), and on @t2 preceding @t1 in the second call's VP
;; metadata (so the backend matches the stale record last).

; REQUIRES: asserts

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc main.ll -o main.o
; RUN: opt -thinlto-bc mid.ll -o mid.o

;; In-process ThinLTO.
; RUN: llvm-lto2 run main.o mid.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=main.o,main,plx \
; RUN:  -r=main.o,t1,plx \
; RUN:  -r=main.o,t2,plx \
; RUN:  -r=main.o,leafA,plx \
; RUN:  -r=main.o,leafB,plx \
; RUN:  -r=main.o,g,plx \
; RUN:  -r=main.o,mid, \
; RUN:  -r=main.o,_Znam, \
; RUN:  -r=mid.o,mid,plx \
; RUN:  -thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats -save-temps \
; RUN:  -o inproc.out 2>&1 | FileCheck %s --check-prefixes=STATS,REMARKS
; RUN: llvm-nm inproc.out.1 | FileCheck %s --check-prefix=NM-MAIN
; RUN: llvm-nm inproc.out.2 | FileCheck %s --check-prefix=NM-MID

; REMARKS-DAG: created clone mid.memprof.1
; REMARKS-DAG: call in clone mid.memprof.1 promoted and assigned to call function clone t2.memprof.1
; REMARKS-DAG: created clone t2.memprof.1
; REMARKS-DAG: created clone leafB.memprof.1

;; @leafA is found through two chains (once for each of its 2 contexts).
; STATS: 2 memprof-context-disambiguation - Number of profiled callees found via multiple tail call chains

; NM-MAIN-DAG: U mid.memprof.1
; NM-MAIN-DAG: T t2.memprof.1
; NM-MAIN-DAG: T leafB.memprof.1

; NM-MID-DAG: T mid{{$}}
; NM-MID-DAG: T mid.memprof.1{{$}}

;; Distributed ThinLTO.
; RUN: llvm-lto2 run main.o mid.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=main.o,main,plx \
; RUN:  -r=main.o,t1,plx \
; RUN:  -r=main.o,t2,plx \
; RUN:  -r=main.o,leafA,plx \
; RUN:  -r=main.o,leafB,plx \
; RUN:  -r=main.o,g,plx \
; RUN:  -r=main.o,mid, \
; RUN:  -r=main.o,_Znam, \
; RUN:  -r=mid.o,mid,plx \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -o distrib.out 2>&1 | FileCheck %s --check-prefix=STATS

;; Only the record for the chain through @t2 is in @mid's summary, with an
;; entry for each of @mid's 2 clones
;; (MD5("mid") = 4395024877212874786, MD5("t2") = 11079814748834529807).
; RUN: llvm-dis mid.o.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX
; INDEX: gv: (guid: 4395024877212874786, summaries: (function: ({{.*}} callsites: ((callee: ^[[T2:[0-9]+]], clones: (0, 1), stackIds: ()))
; INDEX: ^[[T2]] = gv: (guid: 11079814748834529807

; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -summary-file=mid.o.thinlto.bc -memprof-import-summary=mid.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  mid.o -S -o distrib.mid.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-MID
; RUN: FileCheck %s --check-prefix=IR-MID < distrib.mid.ll

; REMARKS-MID-DAG: created clone mid.memprof.1
; REMARKS-MID-DAG: call in clone mid.memprof.1 promoted and assigned to call function clone t2.memprof.1

; IR-MID: define dso_local ptr @mid.memprof.1(
; IR-MID: icmp eq ptr %fp, @t2
; IR-MID: call ptr @t2.memprof.1()

;--- mid.ll
source_filename = "mid.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @mid(ptr %fp, i1 %c) #0 {
entry:
  br i1 %c, label %a, label %b

a:
  %ra = tail call ptr %fp(), !prof !0
  ret ptr %ra

b:
  %rb = tail call ptr %fp(), !prof !1
  ret ptr %rb
}

attributes #0 = { noinline }

;; MD5("t1") = -861406228493635197, MD5("t2") = -7366929324875021809.
!0 = !{!"VP", i32 0, i64 4, i64 -861406228493635197, i64 4}
!1 = !{!"VP", i32 0, i64 10, i64 -7366929324875021809, i64 6, i64 -861406228493635197, i64 4}

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = global i1 false

define i32 @main() {
entry:
  ;; Contexts to @leafB via @t2 (notcold and cold).
  %c1 = call ptr @mid(ptr @t2, i1 false), !callsite !11
  %c2 = call ptr @mid(ptr @t2, i1 false), !callsite !12
  ;; Contexts to @leafA (notcold and cold), reachable via @t1 and @t2.
  %c3 = call ptr @mid(ptr @t1, i1 true), !callsite !13
  %c4 = call ptr @mid(ptr @t1, i1 true), !callsite !14
  ret i32 0
}

declare ptr @mid(ptr, i1)

define ptr @t1() #0 {
entry:
  %r = tail call ptr @leafA()
  ret ptr %r
}

define ptr @t2() #0 {
entry:
  %v = load i1, ptr @g
  br i1 %v, label %a, label %b

a:
  %ra = tail call ptr @leafA()
  ret ptr %ra

b:
  %rb = tail call ptr @leafB()
  ret ptr %rb
}

define ptr @leafA() #0 {
entry:
  %call = tail call ptr @_Znam(i64 10) #1, !memprof !1, !callsite !5
  ret ptr %call
}

define ptr @leafB() #0 {
entry:
  %call = tail call ptr @_Znam(i64 10) #1, !memprof !21, !callsite !25
  ret ptr %call
}

declare ptr @_Znam(i64)

attributes #0 = { noinline }
attributes #1 = { builtin allocsize(0) }

!1 = !{!2, !4}
!2 = !{!3, !"notcold"}
!3 = !{i64 3186456655321080972, i64 1234567890123456789}
!4 = !{!8, !"cold"}
!8 = !{i64 3186456655321080972, i64 2345678901234567890}
!5 = !{i64 3186456655321080972}

!21 = !{!22, !23}
!22 = !{!24, !"notcold"}
!24 = !{i64 4186456655321080972, i64 8632435727821051414}
!23 = !{!28, !"cold"}
!28 = !{i64 4186456655321080972, i64 -3421689549917153178}
!25 = !{i64 4186456655321080972}

!11 = !{i64 8632435727821051414}
!12 = !{i64 -3421689549917153178}
!13 = !{i64 1234567890123456789}
!14 = !{i64 2345678901234567890}
