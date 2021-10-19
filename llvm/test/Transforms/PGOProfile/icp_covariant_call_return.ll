; RUN: opt < %s -pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.D = type { %struct.B }
%struct.B = type { i32 (...)** }
%struct.Base = type { i8 }
%struct.Derived = type { i8 }

declare noalias i8* @_Znwm(i64)
declare void @_ZN1DC2Ev(%struct.D*);
define %struct.Derived* @_ZN1D4funcEv(%struct.D*) {
  ret %struct.Derived* null
}

define %struct.Base* @bar() {
entry:
  %call = call noalias i8* @_Znwm(i64 8)
  %tmp = bitcast i8* %call to %struct.D*
  call void @_ZN1DC2Ev(%struct.D* %tmp)
  %tmp1 = bitcast %struct.D* %tmp to %struct.B*
  %tmp2 = bitcast %struct.B* %tmp1 to %struct.Base* (%struct.B*)***
  %vtable = load %struct.Base* (%struct.B*)**, %struct.Base* (%struct.B*)*** %tmp2, align 8
  %vfn = getelementptr inbounds %struct.Base* (%struct.B*)*, %struct.Base* (%struct.B*)** %vtable, i64 0
  %tmp3 = load %struct.Base* (%struct.B*)*, %struct.Base* (%struct.B*)** %vfn, align 8
; ICALL-PROM:  [[CMP:%[0-9]+]] = icmp eq %struct.Base* (%struct.B*)* %tmp3, bitcast (%struct.Derived* (%struct.D*)* @_ZN1D4funcEv to %struct.Base* (%struct.B*)*)
; ICALL-PROM:  br i1 [[CMP]], label %if.true.direct_targ, label %if.false.orig_indirect, !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICALL-PROM:if.true.direct_targ:
; ICALL-PROM:  [[ARG_BITCAST:%[0-9]+]] = bitcast %struct.B* %tmp1 to %struct.D*
; ICALL-PROM:  [[DIRCALL_RET:%[0-9]+]] = call %struct.Derived* @_ZN1D4funcEv(%struct.D* [[ARG_BITCAST]])
; ICALL-PROM:  [[DIRCALL_RET_CAST:%[0-9]+]] = bitcast %struct.Derived* [[DIRCALL_RET]] to %struct.Base*
; ICALL-PROM:  br label %if.end.icp 
; ICALL-PROM:if.false.orig_indirect:
; ICALL-PROM:  %call1 = call %struct.Base* %tmp3(%struct.B* %tmp1)
; ICALL-PROM:  br label %if.end.icp
; ICALL-PROM:if.end.icp:
; ICALL-PROM:  [[PHI_RET:%[0-9]+]] = phi %struct.Base* [ %call1, %if.false.orig_indirect ], [ [[DIRCALL_RET_CAST]], %if.true.direct_targ ]
  %call1 = call %struct.Base* %tmp3(%struct.B* %tmp1), !prof !1
  ret %struct.Base* %call1
}

!1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
; ICALL-PROM-NOT: !1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
; ICALL-PROM: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 12345, i32 0}
; ICALL-PROM-NOT: !1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
