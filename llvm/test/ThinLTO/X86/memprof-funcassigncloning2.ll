;; Similar to funcassigncloning.ll but hand modified to add another allocation
;; whose pruned cold context only includes an immediate caller node that itself
;; doesn't need cloning, but calls a cloned allocating function, and is in a
;; function that gets cloned multiple times for a different callsite. This test
;; makes sure the non-cloned callsite is correctly updated in all function
;; clones. This case was missed because, due to context pruning, we don't have
;; any caller edges for the first callsite, so the handling that kicks in to
;; "reclone" other callsites in cloned functions was being missed.

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKS

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR


;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -o %t2.out

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=REMARKS


source_filename = "funcassigncloning.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Eventually this function will be cloned several times (for the calls to new
;; for the various callers). However, function blah() includes an allocation
;; whose cold context was trimmed above here. We therefore should assume that
;; every caller of this function should call the same version of blah (which
;; will be the cloned ".memprof.1" version.
; Function Attrs: noinline optnone
define internal void @_Z1EPPcS0_(ptr %buf1, ptr %buf2) #0 {
entry:
  call void @blah(), !callsite !19
  %call = call ptr @_Znam(i64 noundef 10), !memprof !0, !callsite !7
  %call1 = call ptr @_Znam(i64 noundef 10), !memprof !8, !callsite !15
  ret void
}

; REMARKS: call in clone _Z1EPPcS0_ assigned to call function clone blah.memprof.1
; REMARKS: call in clone _Z1EPPcS0_.memprof.1 assigned to call function clone blah.memprof.1
; REMARKS: call in clone _Z1EPPcS0_.memprof.2 assigned to call function clone blah.memprof.1
; REMARKS: call in clone _Z1EPPcS0_.memprof.3 assigned to call function clone blah.memprof.1

; IR: define {{.*}} @_Z1EPPcS0_
; IR:   call {{.*}} @blah.memprof.1()
; IR: define {{.*}} @_Z1EPPcS0_.memprof.2
; IR:   call {{.*}} @blah.memprof.1()
; IR: define {{.*}} @_Z1EPPcS0_.memprof.3
; IR:   call {{.*}} @blah.memprof.1()

declare ptr @_Znam(i64)

define internal void @_Z1BPPcS0_() {
entry:
  call void @_Z1EPPcS0_(ptr null, ptr null), !callsite !16
  ret void
}

define internal void @_Z1CPPcS0_() {
entry:
  call void @_Z1EPPcS0_(ptr null, ptr null), !callsite !17
  ret void
}

define internal void @_Z1DPPcS0_() {
entry:
  call void @_Z1EPPcS0_(ptr null, ptr null), !callsite !18
  ret void
}

; Function Attrs: noinline optnone
define i32 @main() #0 {
entry:
  call void @_Z1BPPcS0_()
  call void @_Z1CPPcS0_()
  call void @_Z1DPPcS0_()
  ret i32 0
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

; uselistorder directives
uselistorder ptr @_Znam, { 1, 0, 2 }

attributes #0 = { noinline optnone }

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
