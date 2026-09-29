; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%struct.D = type { %struct.B }
%struct.B = type { ptr }
%struct.Derived = type { %struct.Base, i32 }
%struct.Base = type { i32 }

@_ZTIi = external constant ptr
declare ptr @_Znwm(i64)
declare void @_ZN1DC2Ev(ptr)
define ptr @_ZN1D4funcEv(ptr) {
  ret ptr null
}
declare void @_ZN1DD0Ev(ptr)
declare void @_ZdlPv(ptr)
declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(ptr)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()


define i32 @foo() personality ptr @__gxx_personality_v0 {
entry:
  %call = invoke ptr @_Znwm(i64 8)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  call void @_ZN1DC2Ev(ptr %call)
  %vtable = load ptr, ptr %call, align 8
  %tmp3 = load ptr, ptr %vtable, align 8
; ICALL-PROM:  [[CMP:%[0-9]+]] = icmp eq ptr %tmp3, @_ZN1D4funcEv
; ICALL-PROM:  br i1 [[CMP]], label %if.true.direct_targ, label %if.false.orig_indirect, !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICALL-PROM:if.true.direct_targ:
; ICALL-PROM:  [[DIRCALL_RET:%[0-9]+]] = invoke ptr @_ZN1D4funcEv(ptr %call)
; ICALL-PROM-NEXT:          to label %if.end.icp unwind label %lpad
; ICALL-PROM:if.false.orig_indirect:
; ICAll-PROM:  %call2 = invoke ptr %tmp3(ptr %call)
; ICAll-PROM:          to label %invoke.cont1 unwind label %lpad
; ICALL-PROM:if.end.icp:
; ICALL-PROM:  br label %invoke.cont1
  %call2 = invoke ptr %tmp3(ptr %call)
          to label %invoke.cont1 unwind label %lpad, !prof !1

invoke.cont1:
; ICAll-PROM:  [[PHI_RET:%[0-9]+]] = phi ptr [ %call2, %if.false.orig_indirect ], [ [[DIRCALL_RET]], %if.end.icp ]
; ICAll-PROM:  %isnull = icmp eq ptr [[PHI_RET]], null
  %isnull = icmp eq ptr %call2, null
  br i1 %isnull, label %delete.end, label %delete.notnull

delete.notnull:
  call void @_ZdlPv(ptr %call2)
  br label %delete.end

delete.end:
  %isnull3 = icmp eq ptr %call, null
  br i1 %isnull3, label %delete.end8, label %delete.notnull4

delete.notnull4:
  %vtable5 = load ptr, ptr %call, align 8
  %vfn6 = getelementptr inbounds ptr, ptr %vtable5, i64 2
  %tmp6 = load ptr, ptr %vfn6, align 8
  invoke void %tmp6(ptr %call)
          to label %invoke.cont7 unwind label %lpad

invoke.cont7:
  br label %delete.end8

delete.end8:
  br label %try.cont

lpad:
  %tmp7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %tmp8 = extractvalue { ptr, i32 } %tmp7, 0
  %tmp9 = extractvalue { ptr, i32 } %tmp7, 1
  br label %catch.dispatch

catch.dispatch:
  %tmp10 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %tmp9, %tmp10
  br i1 %matches, label %catch, label %eh.resume

catch:
  %tmp11 = call ptr @__cxa_begin_catch(ptr %tmp8)
  %tmp13 = load i32, ptr %tmp11, align 4
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret i32 0

eh.resume:
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %tmp8, 0
  %lpad.val11 = insertvalue { ptr, i32 } %lpad.val, i32 %tmp9, 1
  resume { ptr, i32 } %lpad.val11
}

!1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
; ICALL-PROM-NOT: !1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
; ICALL-PROM: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 12345, i32 0}
; ICALL-PROM-NOT: !1 = !{!"VP", i32 0, i64 12345, i64 -3913987384944532146, i64 12345}
