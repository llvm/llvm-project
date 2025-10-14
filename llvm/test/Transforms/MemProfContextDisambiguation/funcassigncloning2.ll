;; Similar to funcassigncloning.ll but hand modified to add another allocation
;; whose pruned cold context only includes an immediate caller node that itself
;; doesn't need cloning, but calls a cloned allocating function, and is in a
;; function that gets cloned multiple times for a different callsite. This test
;; makes sure the non-cloned callsite is correctly updated in all function
;; clones. This case was missed because, due to context pruning, we don't have
;; any caller edges for the first callsite, so the handling that kicks in to
;; "reclone" other callsites in cloned functions was being missed.

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=IR --check-prefix=REMARKS


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Eventually this function will be cloned several times (for the calls to new
;; for the various callers). However, function blah() includes an allocation
;; whose cold context was trimmed above here. We therefore should assume that
;; every caller of this function should call the same version of blah (which
;; will be the cloned ".memprof.1" version.
define internal void @_Z1EPPcS0_(ptr %buf1, ptr %buf2) #0 {
entry:
  call void @blah(), !callsite !19
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !0, !callsite !7
  %call1 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !8, !callsite !15
  ret void
}

; REMARKS: created clone blah.memprof.1
; REMARKS: call in clone _Z1EPPcS0_ assigned to call function clone blah.memprof.1
; REMARKS: call in clone _Z1EPPcS0_.memprof.1 assigned to call function clone blah.memprof.1
; REMARKS: call in clone _Z1EPPcS0_.memprof.2 assigned to call function clone blah.memprof.1
; REMARKS: call in clone _Z1EPPcS0_.memprof.3 assigned to call function clone blah.memprof.1

; IR: define {{.*}} @_Z1EPPcS0_
; IR:   call {{.*}} @blah.memprof.1()
; IR: define {{.*}} @_Z1EPPcS0_.memprof.1
; IR:   call {{.*}} @blah.memprof.1()
; IR: define {{.*}} @_Z1EPPcS0_.memprof.2
; IR:   call {{.*}} @blah.memprof.1()
; IR: define {{.*}} @_Z1EPPcS0_.memprof.3
; IR:   call {{.*}} @blah.memprof.1()

declare ptr @_Znam(i64) #1

define internal void @_Z1BPPcS0_(ptr %0, ptr %1) {
entry:
  call void @_Z1EPPcS0_(ptr noundef %0, ptr noundef %1), !callsite !16
  ret void
}

; Function Attrs: noinline
define internal void @_Z1CPPcS0_(ptr %0, ptr %1) #2 {
entry:
  call void @_Z1EPPcS0_(ptr noundef %0, ptr noundef %1), !callsite !17
  ret void
}

define internal void @_Z1DPPcS0_(ptr %0, ptr %1) #3 {
entry:
  call void @_Z1EPPcS0_(ptr noundef %0, ptr noundef %1), !callsite !18
  ret void
}

define internal void @blah() #0 {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !22, !callsite !21
  ret void
}

define internal void @foo() #0 {
entry:
  call void @blah(), !callsite !20
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

declare i32 @sleep() #5

; uselistorder directives
uselistorder ptr @_Znam, { 1, 0, 2 }

attributes #0 = { "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #1 = { "no-trapping-math"="true" }
attributes #2 = { noinline }
attributes #3 = { "frame-pointer"="all" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #5 = { "disable-tail-calls"="true" }
attributes #6 = { builtin }

!0 = !{!1, !3, !5}
!1 = !{!2, !"cold"}
!2 = !{i64 -3461278137325233666, i64 -7799663586031895603}
!3 = !{!4, !"notcold"}
!4 = !{i64 -3461278137325233666, i64 -3483158674395044949}
!5 = !{!6, !"notcold"}
!6 = !{i64 -3461278137325233666, i64 -2441057035866683071}
!7 = !{i64 -3461278137325233666}
!8 = !{!9, !11, !13}
!9 = !{!10, !"notcold"}
!10 = !{i64 -1415475215210681400, i64 -2441057035866683071}
!11 = !{!12, !"cold"}
!12 = !{i64 -1415475215210681400, i64 -3483158674395044949}
!13 = !{!14, !"notcold"}
!14 = !{i64 -1415475215210681400, i64 -7799663586031895603}
!15 = !{i64 -1415475215210681400}
!16 = !{i64 -2441057035866683071}
!17 = !{i64 -3483158674395044949}
!18 = !{i64 -7799663586031895603}
!19 = !{i64 123}
!20 = !{i64 234}
!21 = !{i64 345}
!22 = !{!23, !25}
!23 = !{!24, !"cold"}
!24 = !{i64 345, i64 123}
!25 = !{!26, !"notcold"}
!26 = !{i64 345, i64 234}
