; RUN: opt < %s -disable-output -jump-threading

%class.E = type { i32 (...)**, %class.C }
%class.C = type { %class.A }
%class.A = type { i32 }
%class.D = type { %class.F }
%class.F = type { %class.E }
%class.B = type { %class.D* }

@_ZTV1D = unnamed_addr constant [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI1D to i8*), i8* bitcast (void (%class.D*)* @_ZN1D7doApplyEv to i8*)]
@_ZTI1D = external unnamed_addr constant { i8*, i8*, i8* }

define void @_ZN15EditCommandImpl5applyEv(%class.E* %this) uwtable align 2 {
entry:
  %0 = bitcast %class.E* %this to void (%class.E*)***
  %vtable = load void (%class.E*)**, void (%class.E*)*** %0, align 8
  %1 = load void (%class.E*)*, void (%class.E*)** %vtable, align 8
  call void %1(%class.E* %this)
  ret void
}

define void @_ZN1DC1Ev(%class.D* nocapture %this) unnamed_addr uwtable align 2 {
entry:
  call void @_ZN24CompositeEditCommandImplC2Ev()
  %0 = getelementptr inbounds %class.D, %class.D* %this, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @_ZTV1D, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

define void @_ZN1DC2Ev(%class.D* nocapture %this) unnamed_addr uwtable align 2 {
entry:
  call void @_ZN24CompositeEditCommandImplC2Ev()
  %0 = getelementptr inbounds %class.D, %class.D* %this, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @_ZTV1D, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

declare void @_ZN24CompositeEditCommandImplC2Ev() #1

define void @_ZN1D7doApplyEv(%class.D* nocapture %this) unnamed_addr nounwind readnone uwtable align 2 {
entry:
  ret void
}

define void @_Z3fn1v() uwtable personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = call noalias i8* @_Znwm() #8
  invoke void @_ZN24CompositeEditCommandImplC2Ev()
          to label %_ZN1DC1Ev.exit unwind label %lpad

_ZN1DC1Ev.exit:                                   ; preds = %entry
  %0 = bitcast i8* %call to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @_ZTV1D, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  %_ref.i.i.i = getelementptr inbounds i8, i8* %call, i64 8
  %1 = bitcast i8* %_ref.i.i.i to i32*
  %2 = load i32, i32* %1, align 4
  %inc.i.i.i = add nsw i32 %2, 1
  store i32 %inc.i.i.i, i32* %1, align 4
  %3 = bitcast i8* %call to %class.D*
  invoke void @_ZN1D7doApplyEv(%class.D* %3)
          to label %_ZN15EditCommandImpl5applyEv.exit unwind label %lpad1

_ZN15EditCommandImpl5applyEv.exit:                ; preds = %_ZN1DC1Ev.exit
  invoke void @_ZN1D16deleteKeyPressedEv()
          to label %invoke.cont7 unwind label %lpad1

invoke.cont7:                                     ; preds = %_ZN15EditCommandImpl5applyEv.exit
  ret void

lpad:                                             ; preds = %entry
  %4 = landingpad { i8*, i32 }
          cleanup
  call void @_ZdlPv() #9
  unreachable

lpad1:                                            ; preds = %_ZN1DC1Ev.exit, %_ZN15EditCommandImpl5applyEv.exit
  %5 = landingpad { i8*, i32 }
          cleanup
  %6 = load i32, i32* %1, align 4
  %tobool.i.i.i = icmp eq i32 %6, 0
  br i1 %tobool.i.i.i, label %_ZN1BI1DED1Ev.exit, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %lpad1
  br i1 undef, label %_ZN1BI1DED1Ev.exit, label %delete.notnull.i.i.i

delete.notnull.i.i.i:                             ; preds = %if.then.i.i.i
  call void @_ZdlPv() #9
  unreachable

_ZN1BI1DED1Ev.exit:                               ; preds = %lpad1, %if.then.i.i.i
  resume { i8*, i32 } undef

terminate.lpad:                                   ; No predecessors!
  %7 = landingpad { i8*, i32 }
          catch i8* null
  unreachable
}

define void @_ZN1BI1DEC1EPS0_(%class.B* nocapture %this, %class.D* %p1) unnamed_addr uwtable align 2 {
entry:
  %m_ptr.i = getelementptr inbounds %class.B, %class.B* %this, i64 0, i32 0
  store %class.D* %p1, %class.D** %m_ptr.i, align 8
  %_ref.i.i = getelementptr inbounds %class.D, %class.D* %p1, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %0 = load i32, i32* %_ref.i.i, align 4
  %inc.i.i = add nsw i32 %0, 1
  store i32 %inc.i.i, i32* %_ref.i.i, align 4
  ret void
}

declare noalias i8* @_Znwm()

declare i32 @__gxx_personality_v0(...)

declare void @_ZdlPv()

define %class.D* @_ZN1BI1DEptEv(%class.B* nocapture readonly %this) nounwind readonly uwtable align 2 {
entry:
  %m_ptr = getelementptr inbounds %class.B, %class.B* %this, i64 0, i32 0
  %0 = load %class.D*, %class.D** %m_ptr, align 8
  ret %class.D* %0
}

declare void @_ZN1D16deleteKeyPressedEv()

define void @_ZN1BI1DED1Ev(%class.B* nocapture readonly %this) unnamed_addr uwtable align 2 {
entry:
  %m_ptr.i = getelementptr inbounds %class.B, %class.B* %this, i64 0, i32 0
  %0 = load %class.D*, %class.D** %m_ptr.i, align 8
  %_ref.i.i = getelementptr inbounds %class.D, %class.D* %0, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %1 = load i32, i32* %_ref.i.i, align 4
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

define void @_ZN1BI1DED2Ev(%class.B* nocapture readonly %this) unnamed_addr uwtable align 2 {
entry:
  %m_ptr = getelementptr inbounds %class.B, %class.B* %this, i64 0, i32 0
  %0 = load %class.D*, %class.D** %m_ptr, align 8
  %_ref.i = getelementptr inbounds %class.D, %class.D* %0, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %1 = load i32, i32* %_ref.i, align 4
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

define void @_ZN1AI1CE5derefEv(%class.A* nocapture readonly %this) nounwind uwtable align 2 {
entry:
  %_ref = getelementptr inbounds %class.A, %class.A* %this, i64 0, i32 0
  %0 = load i32, i32* %_ref, align 4
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

define void @_ZN1BI1DEC2EPS0_(%class.B* nocapture %this, %class.D* %p1) unnamed_addr uwtable align 2 {
entry:
  %m_ptr = getelementptr inbounds %class.B, %class.B* %this, i64 0, i32 0
  store %class.D* %p1, %class.D** %m_ptr, align 8
  %_ref.i = getelementptr inbounds %class.D, %class.D* %p1, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %0 = load i32, i32* %_ref.i, align 4
  %inc.i = add nsw i32 %0, 1
  store i32 %inc.i, i32* %_ref.i, align 4
  ret void
}

define void @_ZN1AI1CE3refEv(%class.A* nocapture %this) nounwind uwtable align 2 {
entry:
  %_ref = getelementptr inbounds %class.A, %class.A* %this, i64 0, i32 0
  %0 = load i32, i32* %_ref, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %_ref, align 4
  ret void
}
