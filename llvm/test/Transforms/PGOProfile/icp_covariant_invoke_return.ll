; RUN: opt < %s -pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%struct.D = type { %struct.B }
%struct.B = type { i32 (...)** }
%struct.Derived = type { %struct.Base, i32 }
%struct.Base = type { i32 }

@_ZTIi = external constant i8*
declare i8* @_Znwm(i64)
declare void @_ZN1DC2Ev(%struct.D*)
declare %struct.Derived* @_ZN1D4funcEv(%struct.D*)
declare void @_ZN1DD0Ev(%struct.D*)
declare void @_ZdlPv(i8*)
declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()


define i32 @foo() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = invoke i8* @_Znwm(i64 8)
          to label %invoke.cont unwind label %lpad

invoke.cont:
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
; ICALL-PROM:  [[DIRCALL_RET:%[0-9]+]] = invoke %struct.Derived* @_ZN1D4funcEv(%struct.D* [[ARG_BITCAST]])
; ICALL-PROM:          to label %if.true.direct_targ.if.end.icp_crit_edge unwind label %lpad
; ICALL-PROM:if.true.direct_targ.if.end.icp_crit_edge:
; ICALL-PROM:  [[DIRCALL_RET_CAST:%[0-9]+]] = bitcast %struct.Derived* [[DIRCALL_RET]] to %struct.Base*
; ICALL-PROM:  br label %if.end.icp
; ICALL-PROM:if.false.orig_indirect:
; ICAll-PROM:  %call2 = invoke %struct.Base* %tmp3(%struct.B* %tmp1)
; ICAll-PROM:          to label %invoke.cont1 unwind label %lpad
; ICALL-PROM:if.end.icp:
; ICALL-PROM:  br label %invoke.cont1
  %call2 = invoke %struct.Base* %tmp3(%struct.B* %tmp1)
          to label %invoke.cont1 unwind label %lpad, !prof !1

invoke.cont1:
; ICAll-PROM:  [[PHI_RET:%[0-9]+]] = phi %struct.Base* [ %call2, %if.false.orig_indirect ], [ [[DIRCALL_RET_CAST]], %if.end.icp ]
; ICAll-PROM:  %isnull = icmp eq %struct.Base* [[PHI_RET]], null
  %isnull = icmp eq %struct.Base* %call2, null
  br i1 %isnull, label %delete.end, label %delete.notnull

delete.notnull:
  %tmp4 = bitcast %struct.Base* %call2 to i8*
  call void @_ZdlPv(i8* %tmp4)
  br label %delete.end

delete.end:
  %isnull3 = icmp eq %struct.B* %tmp1, null
  br i1 %isnull3, label %delete.end8, label %delete.notnull4

delete.notnull4:
  %tmp5 = bitcast %struct.B* %tmp1 to void (%struct.B*)***
  %vtable5 = load void (%struct.B*)**, void (%struct.B*)*** %tmp5, align 8
  %vfn6 = getelementptr inbounds void (%struct.B*)*, void (%struct.B*)** %vtable5, i64 2
  %tmp6 = load void (%struct.B*)*, void (%struct.B*)** %vfn6, align 8
  invoke void %tmp6(%struct.B* %tmp1)
          to label %invoke.cont7 unwind label %lpad

invoke.cont7:
  br label %delete.end8

delete.end8:
  br label %try.cont

lpad:
  %tmp7 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %tmp8 = extractvalue { i8*, i32 } %tmp7, 0
  %tmp9 = extractvalue { i8*, i32 } %tmp7, 1
  br label %catch.dispatch

catch.dispatch:
  %tmp10 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %tmp9, %tmp10
  br i1 %matches, label %catch, label %eh.resume

catch:
  %tmp11 = call i8* @__cxa_begin_catch(i8* %tmp8)
  %tmp12 = bitcast i8* %tmp11 to i32*
  %tmp13 = load i32, i32* %tmp12, align 4
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret i32 0

eh.resume:
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %tmp8, 0
  %lpad.val11 = insertvalue { i8*, i32 } %lpad.val, i32 %tmp9, 1
  resume { i8*, i32 } %lpad.val11
}

!1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
; ICALL-PROM-NOT: !1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
; ICALL-PROM: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 12345, i32 0}
; ICALL-PROM-NOT: !1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
