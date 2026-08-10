;; This test ensures that the logic which assigns calls to stack nodes
;; correctly handles a case where multiple nodes have stack ids that
;; overlap with each other but have different last nodes (can happen with
;; inlining into various levels of a call chain). Specifically, when we
;; have one that is duplicated (e.g. unrolling), we need to correctly
;; handle the case where the context id has already been assigned to
;; a different call with a different last node.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS --check-prefix=REMARKS

; REMARKS: created clone _Z1Ab.memprof.1
; REMARKS: created clone _Z3XZNv.memprof.1
; REMARKS: call in clone main assigned to call function clone _Z3XZNv.memprof.1
;; Make sure the inlined context in _Z3XZNv, which partially overlaps the stack
;; ids in the shorter inlined context of Z2XZv, correctly calls a cloned
;; version of Z1Ab, which will call the cold annotated allocation.
; REMARKS: call in clone _Z3XZNv.memprof.1 assigned to call function clone _Z1Ab.memprof.1
; REMARKS: call in clone _Z1Ab.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone main assigned to call function clone _Z3XZNv
; REMARKS: call in clone _Z3XZNv assigned to call function clone _Z1Ab
; REMARKS: call in clone _Z1Ab marked with memprof allocation attribute notcold


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z1Ab() {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #1, !memprof !0, !callsite !5
  ret void
}

; Function Attrs: nobuiltin
declare ptr @_Znam(i64) #0

;; Inlining of stack id 2 into 3. Assume this is called from somewhere else.
define dso_local void @_Z2XZv() local_unnamed_addr #0 {
entry:
  ;; Simulate duplication of the callsite (e.g. unrolling).
  call void @_Z1Ab(), !callsite !6
  call void @_Z1Ab(), !callsite !6
  ret void
}

;; Inlining of stack id 2 into 3 into 4. Called by main below.
define dso_local void @_Z3XZNv() local_unnamed_addr {
entry:
  call void @_Z1Ab(), !callsite !7
  ret void
}

define dso_local noundef i32 @main() local_unnamed_addr {
entry:
  call void @_Z3XZNv(), !callsite !8 ;; Not cold context
  call void @_Z3XZNv(), !callsite !9 ;; Cold context
  ret i32 0
}

attributes #0 = { nobuiltin }
attributes #7 = { builtin }

!0 = !{!1, !3}
;; Not cold context via first call to _Z3XZNv in main
!1 = !{!2, !"notcold"}
!2 = !{i64 1, i64 2, i64 3, i64 4, i64 5}
;; Cold context via second call to _Z3XZNv in main
!3 = !{!4, !"cold"}
!4 = !{i64 1, i64 2, i64 3, i64 4, i64 6}
!5 = !{i64 1}
!6 = !{i64 2, i64 3}
!7 = !{i64 2, i64 3, i64 4}
!8 = !{i64 5}
!9 = !{i64 6}

; IR: define {{.*}} @_Z1Ab()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR: define {{.*}} @_Z2XZv()
; IR:   call {{.*}} @_Z1Ab()
; IR:   call {{.*}} @_Z1Ab()
; IR: define {{.*}} @_Z3XZNv()
; IR:   call {{.*}} @_Z1Ab()
; IR: define {{.*}} @main()
; IR:   call {{.*}} @_Z3XZNv()
; IR:   call {{.*}} @_Z3XZNv.memprof.1()
; IR: define {{.*}} @_Z1Ab.memprof.1()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR: define {{.*}} @_Z3XZNv.memprof.1()
; IR:   call {{.*}} @_Z1Ab.memprof.1()

; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }

; STATS: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 2 memprof-context-disambiguation - Number of function clones created during whole program analysis
