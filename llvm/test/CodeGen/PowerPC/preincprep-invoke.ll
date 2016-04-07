; RUN: llc -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@.str1 = external unnamed_addr constant [1 x i8], align 1
@.str2 = external unnamed_addr constant [39 x i8], align 1

declare void @_ZN13CStdOutStreamlsEPKc()

declare void @_ZN13CStdOutStream5FlushEv()

declare i32 @__gxx_personality_v0(...)

define void @_Z11GetPasswordP13CStdOutStreamb(i1 %cond, i8 %arg1, i8* %arg2) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.cond.i.i, %entry
  br i1 %cond, label %_ZN11CStringBaseIcEC2EPKc.exit.critedge, label %for.cond.i.i

_ZN11CStringBaseIcEC2EPKc.exit.critedge:          ; preds = %for.cond.i.i
  invoke void @_ZN13CStdOutStreamlsEPKc()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %_ZN11CStringBaseIcEC2EPKc.exit.critedge
  invoke void @_ZN13CStdOutStream5FlushEv()
          to label %invoke.cont4 unwind label %lpad

invoke.cont4:                                     ; preds = %invoke.cont
  %call7 = invoke i8* @getpass()
          to label %for.cond.i.i30 unwind label %lpad

; CHECK-LABEL: @_Z11GetPasswordP13CStdOutStreamb
; CHECK: addi {{[0-9]+}}, 3, -1

for.cond.i.i30:                                   ; preds = %for.cond.i.i30, %invoke.cont4
  %indvars.iv.i.i26 = phi i64 [ %indvars.iv.next.i.i29, %for.cond.i.i30 ], [ 0, %invoke.cont4 ]
  %arrayidx.i.i27 = getelementptr inbounds i8, i8* %call7, i64 %indvars.iv.i.i26
  %0 = load i8, i8* %arrayidx.i.i27, align 1
  %1 = add i8 %0, %arg1
  store i8 %1, i8* %arg2, align 1
  %indvars.iv.next.i.i29 = add nuw nsw i64 %indvars.iv.i.i26, 1
  br label %for.cond.i.i30

lpad:                                             ; preds = %invoke.cont4, %invoke.cont, %_ZN11CStringBaseIcEC2EPKc.exit.critedge
  %2 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

declare i8* @getpass()

