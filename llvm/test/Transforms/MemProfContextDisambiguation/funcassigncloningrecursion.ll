;; Test context disambiguation for a callgraph containing multiple memprof
;; contexts with recursion, where we need to perform additional cloning
;; during function assignment/cloning to handle the combination of contexts
;; to 2 different allocations. In particular, ensures that the recursive edges
;; are handled correctly during the function assignment cloning, where they
;; were previously causing an assert (and infinite recursion in an NDEBUG
;; compile).
;;
;; This test is a hand modified version of funcassigncloning.ll, where all
;; the calls to new were moved into one function, with several recursive
;; calls for the different contexts. The code as written here is somewhat
;; nonsensical as it would produce infinite recursion, but in a real case
;; the recursive calls would be suitably guarded.
;;
;; For this test we just ensure that it doesn't crash, and gets the expected
;; cloning remarks.

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=REMARKS


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_Znam(i64) #1

define internal void @_Z1DPPcS0_(ptr %0, ptr %1) #3 {
entry:
  call void @_Z1DPPcS0_(ptr noundef %0, ptr noundef %1), !callsite !16
  call void @_Z1DPPcS0_(ptr noundef %0, ptr noundef %1), !callsite !17
  call void @_Z1DPPcS0_(ptr noundef %0, ptr noundef %1), !callsite !18
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !0, !callsite !7
  %call1 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !8, !callsite !15
  ret void
}

; uselistorder directives
uselistorder ptr @_Znam, { 1, 0 }

attributes #1 = { "no-trapping-math"="true" }
attributes #3 = { "frame-pointer"="all" }
attributes #6 = { builtin }

!0 = !{!1, !3, !5}
!1 = !{!2, !"cold"}
!2 = !{i64 -3461278137325233666, i64 -7799663586031895603, i64 -7799663586031895603}
!3 = !{!4, !"notcold"}
!4 = !{i64 -3461278137325233666, i64 -3483158674395044949, i64 -3483158674395044949}
!5 = !{!6, !"notcold"}
!6 = !{i64 -3461278137325233666, i64 -2441057035866683071, i64 -2441057035866683071}
!7 = !{i64 -3461278137325233666}
!8 = !{!9, !11, !13}
!9 = !{!10, !"notcold"}
!10 = !{i64 -1415475215210681400, i64 -2441057035866683071, i64 -2441057035866683071}
!11 = !{!12, !"cold"}
!12 = !{i64 -1415475215210681400, i64 -3483158674395044949, i64 -3483158674395044949}
!13 = !{!14, !"notcold"}
!14 = !{i64 -1415475215210681400, i64 -7799663586031895603, i64 -7799663586031895603}
!15 = !{i64 -1415475215210681400}
!16 = !{i64 -2441057035866683071}
!17 = !{i64 -3483158674395044949}
!18 = !{i64 -7799663586031895603}


;; We greedily create a clone that is initially used by the clones of the
;; first call to new. However, we end up with an incompatible set of callers
;; given the second call to new which has clones with a different combination of
;; callers. Eventually, we create 2 more clones, and the first clone becomes dead.
; REMARKS: created clone _Z1DPPcS0_.memprof.1
; REMARKS: created clone _Z1DPPcS0_.memprof.2
; REMARKS: created clone _Z1DPPcS0_.memprof.3
; REMARKS: call in clone _Z1DPPcS0_ assigned to call function clone _Z1DPPcS0_.memprof.2
; REMARKS: call in clone _Z1DPPcS0_.memprof.2 marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1DPPcS0_ assigned to call function clone _Z1DPPcS0_.memprof.3
; REMARKS: call in clone _Z1DPPcS0_.memprof.3 marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1DPPcS0_ assigned to call function clone _Z1DPPcS0_
; REMARKS: call in clone _Z1DPPcS0_ marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1DPPcS0_.memprof.2 marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1DPPcS0_.memprof.3 marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1DPPcS0_ marked with memprof allocation attribute notcold
