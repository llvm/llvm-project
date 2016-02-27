; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @updateModelQPFrame(i32 %m_Bits) {
entry:
  %0 = load double, double* undef, align 8
  %mul = fmul double undef, %0
  %mul2 = fmul double undef, %mul
  %mul4 = fmul double %0, %mul2
  %mul5 = fmul double undef, 4.000000e+00
  %mul7 = fmul double undef, %mul5
  %conv = sitofp i32 %m_Bits to double
  %mul8 = fmul double %conv, %mul7
  %add = fadd double %mul4, %mul8
  %cmp11 = fcmp olt double %add, 0.000000e+00
  ret void
}

declare i8* @objc_msgSend(i8*, i8*, ...)
declare i32 @personality_v0(...)

define void @invoketest() personality i8* bitcast (i32 (...)* @personality_v0 to i8*) {
entry:
  br i1 undef, label %cond.true, label %cond.false

cond.true:
  %call49 = invoke double bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to double (i8*, i8*)*)(i8* undef, i8* undef) 
          to label %cond.true54 unwind label %lpad

cond.false:
  %call51 = invoke double bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to double (i8*, i8*)*)(i8* undef, i8* undef)
          to label %cond.false57 unwind label %lpad

cond.true54:
  %call56 = invoke double bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to double (i8*, i8*)*)(i8* undef, i8* undef) 
          to label %cond.end60 unwind label %lpad

cond.false57:
  %call59 = invoke double bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to double (i8*, i8*)*)(i8* undef, i8* undef)
          to label %cond.end60 unwind label %lpad

; Make sure we don't vectorize these phis - they have invokes as inputs.

; RUN: opt < %s -slp-vectorizer -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7 | FileCheck %s

; CHECK-LABEL: invoketest

; CHECK-LABEL: cond.end60
; CHECK-NOT: phi <2 x double>
; CHECK: insertelement
; CHECK-LABEL: if.then63

cond.end60:
  %cond126 = phi double [ %call49, %cond.true54 ], [ %call51, %cond.false57 ]
  %cond61 = phi double [ %call56, %cond.true54 ], [ %call59, %cond.false57 ]
  br i1 undef, label %if.end98, label %if.then63

if.then63:
  %conv69 = fptrunc double undef to float
  %conv70 = fpext float %conv69 to double
  %div71 = fdiv double %cond126, %conv70
  %conv78 = fptrunc double undef to float
  %conv79 = fpext float %conv78 to double
  %div80 = fdiv double %cond61, %conv79
  br label %if.end98

lpad:
  %l = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %l

if.end98:
  %dimensionsResult.sroa.0.0 = phi double [ %div71, %if.then63 ], [ %cond126, %cond.end60 ]
  %dimensionsResult.sroa.6.0 = phi double [ %div80, %if.then63 ], [ %cond61, %cond.end60 ]
  br label %if.end99

if.end99:
  ret void
}
