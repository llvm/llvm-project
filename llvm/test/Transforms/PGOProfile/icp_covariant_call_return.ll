; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.D = type { %struct.B }
%struct.B = type { ptr }
%struct.Base = type { i8 }
%struct.Derived = type { i8 }

declare noalias ptr @_Znwm(i64)
declare void @_ZN1DC2Ev(ptr);
define ptr @_ZN1D4funcEv(ptr) {
  ret ptr null
}

define ptr @bar() {
entry:
  %call = call noalias ptr @_Znwm(i64 8)
  call void @_ZN1DC2Ev(ptr %call)
  %vtable = load ptr, ptr %call, align 8
  %tmp3 = load ptr, ptr %vtable, align 8
; ICALL-PROM:  [[CMP:%[0-9]+]] = icmp eq ptr %tmp3, @_ZN1D4funcEv
; ICALL-PROM:  br i1 [[CMP]], label %if.true.direct_targ, label %if.false.orig_indirect, !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICALL-PROM:if.true.direct_targ:
; ICALL-PROM:  [[DIRCALL_RET:%[0-9]+]] = call ptr @_ZN1D4funcEv(ptr %call)
; ICALL-PROM:  br label %if.end.icp 
; ICALL-PROM:if.false.orig_indirect:
; ICALL-PROM:  %call1 = call ptr %tmp3(ptr %call)
; ICALL-PROM:  br label %if.end.icp
; ICALL-PROM:if.end.icp:
; ICALL-PROM:  [[PHI_RET:%[0-9]+]] = phi ptr [ %call1, %if.false.orig_indirect ], [ [[DIRCALL_RET]], %if.true.direct_targ ]
  %call1 = call ptr %tmp3(ptr %call), !prof !1
  ret ptr %call1
}

!1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
; ICALL-PROM-NOT: !1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
; ICALL-PROM: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 12345, i32 0}
; ICALL-PROM-NOT: !1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
