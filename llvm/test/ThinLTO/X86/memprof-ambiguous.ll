;; The -stats checks need assertions.
; REQUIRES: asserts

;; Test that an allocation whose cold and not cold contexts cannot be
;; disambiguated by cloning keeps the "ambiguous" memprof attribute, in both
;; regular LTO (opt pass on IR) and ThinLTO (whole program analysis on the
;; summary followed by the backend), and that -memprof-ambiguous-attributes=false
;; restores the previous behavior of hinting such allocations not cold.
;;
;; The callsite metadata needed for cloning is missing from main, so the two
;; contexts (one cold, one not cold) stay on the single allocation in foo.

;; Regular LTO: the pass keeps the ambiguous attribute that matching added.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -pass-remarks=memprof-context-disambiguation -stats %s -S 2>&1 \
; RUN:  | FileCheck %s --check-prefix=IR --check-prefix=REMARKS \
; RUN:  --check-prefix=STATS --implicit-check-not "\"memprof\"=\"notcold\""

;; With ambiguous attributes disabled the allocation gets the not cold default.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-ambiguous-attributes=false \
; RUN:  -pass-remarks=memprof-context-disambiguation %s -S 2>&1 \
; RUN:  | FileCheck %s --check-prefix=IRNOTCOLD --check-prefix=REMARKSNOTCOLD \
; RUN:  --implicit-check-not "\"memprof\"=\"ambiguous\""

;; ThinLTO: the combined allocation type is recorded in the summary, printed as
;; "ambiguous", and the backend keeps the ambiguous attribute.
; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -stats -o %t.out 2>&1 | FileCheck %s --check-prefix=STATS
; RUN: llvm-dis %t.o.thinlto.bc -o - | FileCheck %s --check-prefix=SUMMARY

;; The summary with the ambiguous version round trips through the assembler.
; RUN: llvm-dis %t.o.thinlto.bc -o - | llvm-as -o - | llvm-dis -o - \
; RUN:  | FileCheck %s --check-prefix=SUMMARY

;; ThinLTO backend (distributed).
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation -stats %t.o -S 2>&1 \
; RUN:  | FileCheck %s --check-prefix=IR --check-prefix=REMARKS \
; RUN:  --check-prefix=STATS-BE --implicit-check-not "\"memprof\"=\"notcold\""

;; ThinLTO backend with ambiguous attributes disabled during the whole program
;; analysis: the summary records not cold, which the backend applies.
; RUN: cp %t.o %t2.o
; RUN: llvm-lto2 run %t2.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new -memprof-ambiguous-attributes=false \
; RUN:  -r=%t2.o,main,plx \
; RUN:  -r=%t2.o,_Znam, \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -o %t2.out 2>&1
; RUN: llvm-dis %t2.o.thinlto.bc -o - | FileCheck %s --check-prefix=SUMMARYNOTCOLD
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t2.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation %t2.o -S 2>&1 \
; RUN:  | FileCheck %s --check-prefix=IRNOTCOLD --check-prefix=REMARKSNOTCOLD \
; RUN:  --implicit-check-not "\"memprof\"=\"ambiguous\""

source_filename = "memprof-ambiguous.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  ;; Missing callsite metadata blocks cloning
  %call = call ptr @_Z3foov()
  %call1 = call ptr @_Z3foov()
  ret i32 0
}

define internal ptr @_Z3foov() #0 {
entry:
  ;; Matching adds the ambiguous attribute to allocations with contexts of both
  ;; types.
  %call = call ptr @_Znam(i64 0) #1, !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

; uselistorder directives
uselistorder ptr @_Z3foov, { 1, 0 }

attributes #0 = { noinline optnone }
attributes #1 = { "memprof"="ambiguous" }

!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}

; REMARKS: call in clone _Z3foov marked with memprof allocation attribute ambiguous
; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Znam(i64 0) #[[AMBIGUOUS:[0-9]+]]
; IR: attributes #[[AMBIGUOUS]] = { "memprof"="ambiguous" }
; STATS: 1 memprof-context-disambiguation - Number of ambiguous static allocations (possibly cloned) during whole program analysis
; STATS-BE: 1 memprof-context-disambiguation - Number of ambiguous static allocations (possibly cloned) during ThinLTO backend

; SUMMARY: allocs: ((versions: (ambiguous), memProf: ((type: notcold, stackIds: (8632435727821051414)), (type: cold, stackIds: (15025054523792398438))))
; SUMMARYNOTCOLD: allocs: ((versions: (notcold), memProf: ((type: notcold, stackIds: (8632435727821051414)), (type: cold, stackIds: (15025054523792398438))))

; REMARKSNOTCOLD: call in clone _Z3foov marked with memprof allocation attribute notcold
; IRNOTCOLD: define internal {{.*}} @_Z3foov()
; IRNOTCOLD:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IRNOTCOLD: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
