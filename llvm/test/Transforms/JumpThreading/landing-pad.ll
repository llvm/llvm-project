; RUN: opt -passes=jump-threading -disable-output < %s

%class.E = type { ptr, %class.C }
%class.C = type { %class.A }
%class.A = type { i32 }
%class.D = type { %class.F }
%class.F = type { %class.E }
%class.B = type { ptr }

@_ZTV1D = unnamed_addr constant [3 x ptr] [ptr null, ptr @_ZTI1D, ptr @_ZN1D7doApplyEv]
@_ZTI1D = external unnamed_addr constant { ptr, ptr, ptr }

define void @_ZN15EditCommandImpl5applyEv(ptr %this) uwtable align 2 {
entry:
  %vtable = load ptr, ptr %this, align 8
  %0 = load ptr, ptr %vtable, align 8
  call void %0(ptr %this)
  ret void
}

define void @_ZN1DC1Ev(ptr nocapture %this) unnamed_addr uwtable align 2 {
entry:
  call void @_ZN24CompositeEditCommandImplC2Ev()
  store ptr getelementptr inbounds ([3 x ptr], ptr @_ZTV1D, i64 0, i64 2), ptr %this, align 8
  ret void
}

define void @_ZN1DC2Ev(ptr nocapture %this) unnamed_addr uwtable align 2 {
entry:
  call void @_ZN24CompositeEditCommandImplC2Ev()
  store ptr getelementptr inbounds ([3 x ptr], ptr @_ZTV1D, i64 0, i64 2), ptr %this, align 8
  ret void
}

declare void @_ZN24CompositeEditCommandImplC2Ev() #1

define void @_ZN1D7doApplyEv(ptr nocapture %this) unnamed_addr nounwind readnone uwtable align 2 {
entry:
  ret void
}

define void @_Z3fn1v() uwtable personality ptr @__gxx_personality_v0 {
entry:
  %call = call noalias ptr @_Znwm() #8
  invoke void @_ZN24CompositeEditCommandImplC2Ev()
          to label %_ZN1DC1Ev.exit unwind label %lpad

_ZN1DC1Ev.exit:                                   ; preds = %entry
  store ptr getelementptr inbounds ([3 x ptr], ptr @_ZTV1D, i64 0, i64 2), ptr %call, align 8
  %_ref.i.i.i = getelementptr inbounds i8, ptr %call, i64 8
  %0 = load i32, ptr %_ref.i.i.i, align 4
  %inc.i.i.i = add nsw i32 %0, 1
  store i32 %inc.i.i.i, ptr %_ref.i.i.i, align 4
  invoke void @_ZN1D7doApplyEv(ptr %call)
          to label %_ZN15EditCommandImpl5applyEv.exit unwind label %lpad1

_ZN15EditCommandImpl5applyEv.exit:                ; preds = %_ZN1DC1Ev.exit
  invoke void @_ZN1D16deleteKeyPressedEv()
          to label %invoke.cont7 unwind label %lpad1

invoke.cont7:                                     ; preds = %_ZN15EditCommandImpl5applyEv.exit
  ret void

lpad:                                             ; preds = %entry
  %1 = landingpad { ptr, i32 }
          cleanup
  call void @_ZdlPv() #9
  unreachable

lpad1:                                            ; preds = %_ZN1DC1Ev.exit, %_ZN15EditCommandImpl5applyEv.exit
  %2 = landingpad { ptr, i32 }
          cleanup
  %3 = load i32, ptr %_ref.i.i.i, align 4
  %tobool.i.i.i = icmp eq i32 %3, 0
  br i1 %tobool.i.i.i, label %_ZN1BI1DED1Ev.exit, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %lpad1
  br i1 undef, label %_ZN1BI1DED1Ev.exit, label %delete.notnull.i.i.i

delete.notnull.i.i.i:                             ; preds = %if.then.i.i.i
  call void @_ZdlPv() #9
  unreachable

_ZN1BI1DED1Ev.exit:                               ; preds = %lpad1, %if.then.i.i.i
  resume { ptr, i32 } undef

terminate.lpad:                                   ; No predecessors!
  %4 = landingpad { ptr, i32 }
          catch ptr null
  unreachable
}

define void @_ZN1BI1DEC1EPS0_(ptr nocapture %this, ptr %p1) unnamed_addr uwtable align 2 {
entry:
  store ptr %p1, ptr %this, align 8
  %_ref.i.i = getelementptr inbounds %class.D, ptr %p1, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %0 = load i32, ptr %_ref.i.i, align 4
  %inc.i.i = add nsw i32 %0, 1
  store i32 %inc.i.i, ptr %_ref.i.i, align 4
  ret void
}

declare noalias ptr @_Znwm()

declare i32 @__gxx_personality_v0(...)

declare void @_ZdlPv()

define ptr @_ZN1BI1DEptEv(ptr nocapture readonly %this) nounwind readonly uwtable align 2 {
entry:
  %0 = load ptr, ptr %this, align 8
  ret ptr %0
}

declare void @_ZN1D16deleteKeyPressedEv()

define void @_ZN1BI1DED1Ev(ptr nocapture readonly %this) unnamed_addr uwtable align 2 {
entry:
  %0 = load ptr, ptr %this, align 8
  %_ref.i.i = getelementptr inbounds %class.D, ptr %0, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %1 = load i32, ptr %_ref.i.i, align 4
  %tobool.i.i = icmp eq i32 %1, 0
  br i1 %tobool.i.i, label %_ZN1BI1DED2Ev.exit, label %if.then.i.i

if.then.i.i:                                      ; preds = %entry
  br i1 undef, label %_ZN1BI1DED2Ev.exit, label %delete.notnull.i.i

delete.notnull.i.i:                               ; preds = %if.then.i.i
  call void @_ZdlPv() #9
  unreachable

_ZN1BI1DED2Ev.exit:                               ; preds = %entry, %if.then.i.i
  ret void
}

declare hidden void @__clang_call_terminate()

define void @_ZN1BI1DED2Ev(ptr nocapture readonly %this) unnamed_addr uwtable align 2 {
entry:
  %0 = load ptr, ptr %this, align 8
  %_ref.i = getelementptr inbounds %class.D, ptr %0, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %1 = load i32, ptr %_ref.i, align 4
  %tobool.i = icmp eq i32 %1, 0
  br i1 %tobool.i, label %_ZN1AI1CE5derefEv.exit, label %if.then.i

if.then.i:                                        ; preds = %entry
  br i1 undef, label %_ZN1AI1CE5derefEv.exit, label %delete.notnull.i

delete.notnull.i:                                 ; preds = %if.then.i
  call void @_ZdlPv() #9
  unreachable

_ZN1AI1CE5derefEv.exit:                           ; preds = %entry, %if.then.i
  ret void
}

define void @_ZN1AI1CE5derefEv(ptr nocapture readonly %this) nounwind uwtable align 2 {
entry:
  %0 = load i32, ptr %this, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br i1 undef, label %if.end, label %delete.notnull

delete.notnull:                                   ; preds = %if.then
  call void @_ZdlPv() #9
  unreachable

if.end:                                           ; preds = %entry, %if.then
  ret void
}

define void @_ZN1BI1DEC2EPS0_(ptr nocapture %this, ptr %p1) unnamed_addr uwtable align 2 {
entry:
  store ptr %p1, ptr %this, align 8
  %_ref.i = getelementptr inbounds %class.D, ptr %p1, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %0 = load i32, ptr %_ref.i, align 4
  %inc.i = add nsw i32 %0, 1
  store i32 %inc.i, ptr %_ref.i, align 4
  ret void
}

define void @_ZN1AI1CE3refEv(ptr nocapture %this) nounwind uwtable align 2 {
entry:
  %0 = load i32, ptr %this, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %this, align 4
  ret void
}
