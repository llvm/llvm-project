; RUN: opt < %s -sroa -loop-unswitch -enable-new-pm=0 -disable-output
; RUN: opt < %s -sroa -loop-unswitch -enable-new-pm=0 -enable-mssa-loop-dependency=true -verify-memoryssa -disable-output
; PR11016
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.2"

%class.MyContainer.1.3.19.29 = type { [6 x %class.MyMemVarClass.0.2.18.28*] }
%class.MyMemVarClass.0.2.18.28 = type { i32 }

define void @_ZN11MyContainer1fEi(%class.MyContainer.1.3.19.29* %this, i32 %doit) uwtable ssp align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %inc1 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %conv = sext i32 %inc1 to i64
  %cmp = icmp ult i64 %conv, 6
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tobool = icmp ne i32 %doit, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %idxprom = sext i32 %inc1 to i64
  %array_ = getelementptr inbounds %class.MyContainer.1.3.19.29, %class.MyContainer.1.3.19.29* %this, i32 0, i32 0
  %arrayidx = getelementptr inbounds [6 x %class.MyMemVarClass.0.2.18.28*], [6 x %class.MyMemVarClass.0.2.18.28*]* %array_, i32 0, i64 %idxprom
  %tmp4 = load %class.MyMemVarClass.0.2.18.28*, %class.MyMemVarClass.0.2.18.28** %arrayidx, align 8
  %isnull = icmp eq %class.MyMemVarClass.0.2.18.28* %tmp4, null
  br i1 %isnull, label %for.inc, label %delete.notnull

delete.notnull:                                   ; preds = %if.then
  invoke void @_ZN13MyMemVarClassD1Ev(%class.MyMemVarClass.0.2.18.28* %tmp4)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %delete.notnull
  %0 = bitcast %class.MyMemVarClass.0.2.18.28* %tmp4 to i8*
  call void @_ZdlPv(i8* %0) nounwind
  br label %for.inc

lpad:                                             ; preds = %delete.notnull
  %1 = landingpad { i8*, i32 }
          cleanup
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = extractvalue { i8*, i32 } %1, 1
  %4 = bitcast %class.MyMemVarClass.0.2.18.28* %tmp4 to i8*
  call void @_ZdlPv(i8* %4) nounwind
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %2, 0
  %lpad.val7 = insertvalue { i8*, i32 } %lpad.val, i32 %3, 1
  resume { i8*, i32 } %lpad.val7

for.inc:                                          ; preds = %invoke.cont, %if.then, %for.body
  %inc = add nsw i32 %inc1, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @_ZN13MyMemVarClassD1Ev(%class.MyMemVarClass.0.2.18.28*)

declare i32 @__gxx_personality_v0(...)

declare void @_ZdlPv(i8*) nounwind
