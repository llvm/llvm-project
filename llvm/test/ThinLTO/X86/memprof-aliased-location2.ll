;; Test to ensure a call to a different callee but with the same debug info
;; (and therefore callsite metadata) as a subsequent call in the alloc context
;; does not cause missing or incorrect cloning. This test is otherwise the same
;; as memprof-basic.ll.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,blah, \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes \
; RUN:	-stats -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:	-o %t.out 2>&1 | FileCheck %s \
; RUN:	--check-prefix=STATS --check-prefix=STATS-BE --check-prefix=REMARKS

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR

source_filename = "memprof-aliased-location2.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call ptr @_Z3foov(), !callsite !0
  %call1 = call ptr @_Z3foov(), !callsite !1
  ret i32 0
}

declare void @blah()

define internal ptr @_Z3barv() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3bazv() #0 {
entry:
  %call = call ptr @_Z3barv(), !callsite !8
  ;; Subsequent call to another callee but with the same debug location / callsite id
  call void @blah(), !callsite !8
  ret ptr null
}

define internal ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Z3bazv(), !callsite !9
  ret ptr null
}

; uselistorder directives
uselistorder ptr @_Z3foov, { 1, 0 }

attributes #0 = { noinline optnone }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 -5964873800580613432}
!9 = !{i64 2732490490862098848}

; REMARKS: call in clone main assigned to call function clone _Z3foov.memprof.1
; REMARKS: created clone _Z3barv.memprof.1
; REMARKS: call in clone _Z3barv marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z3barv.memprof.1 marked with memprof allocation attribute cold
; REMARKS: created clone _Z3bazv.memprof.1
; REMARKS: call in clone _Z3bazv.memprof.1 assigned to call function clone _Z3barv.memprof.1
; REMARKS: created clone _Z3foov.memprof.1
; REMARKS: call in clone _Z3foov.memprof.1 assigned to call function clone _Z3bazv.memprof.1


; IR: define {{.*}} @main
;; The first call to foo does not allocate cold memory. It should call the
;; original functions, which ultimately call the original allocation decorated
;; with a "notcold" attribute.
; IR:   call {{.*}} @_Z3foov()
;; The second call to foo allocates cold memory. It should call cloned functions
;; which ultimately call a cloned allocation decorated with a "cold" attribute.
; IR:   call {{.*}} @_Z3foov.memprof.1()
; IR: define internal {{.*}} @_Z3barv()
; IR:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3bazv()
; IR:   call {{.*}} @_Z3barv()
; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Z3bazv()
; IR: define internal {{.*}} @_Z3barv.memprof.1()
; IR:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3bazv.memprof.1()
; IR:   call {{.*}} @_Z3barv.memprof.1()
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Z3bazv.memprof.1()
; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }


; STATS: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS-BE: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned) during ThinLTO backend
; STATS: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS-BE: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned) during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Number of allocation versions (including clones) during ThinLTO backend
; STATS: 3 memprof-context-disambiguation - Number of function clones created during whole program analysis
; STATS-BE: 3 memprof-context-disambiguation - Number of function clones created during ThinLTO backend
; STATS-BE: 3 memprof-context-disambiguation - Number of functions that had clones created during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Maximum number of allocation versions created for an original allocation during ThinLTO backend
; STATS-BE: 1 memprof-context-disambiguation - Number of original (not cloned) allocations with memprof profiles during ThinLTO backend
