target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc < %s | FileCheck %s

define fastcc void @allocateSpace(i1 %cond1, i1 %cond2) nounwind {
entry:
  %0 = load i8** undef, align 8, !tbaa !0
  br i1 undef, label %return, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  br i1 undef, label %if.end7, label %return

if.end7:                                          ; preds = %lor.lhs.false
  br i1 undef, label %if.then15, label %if.end71

if.then15:                                        ; preds = %if.end7
  br label %while.cond

while.cond:                                       ; preds = %while.body, %if.then15
  %idxprom17 = sext i32 0 to i64
  %arrayidx18 = getelementptr inbounds i8* %0, i64 %idxprom17
  %or = or i32 undef, undef
  br i1 %cond1, label %if.end71, label %while.body

while.body:                                       ; preds = %while.cond
  br i1 %cond2, label %while.cond, label %if.then45

if.then45:                                        ; preds = %while.body
  %idxprom48139 = zext i32 %or to i64
  %arrayidx49 = getelementptr inbounds i8* %0, i64 %idxprom48139
  %1 = bitcast i8* %arrayidx49 to i16*
  %2 = bitcast i8* %arrayidx18 to i16*
  %3 = load i16* %1, align 1
  store i16 %3, i16* %2, align 1
  br label %return

if.end71:                                         ; preds = %while.cond, %if.end7
  unreachable

return:                                           ; preds = %if.then45, %lor.lhs.false, %entry
  ret void

; CHECK: @allocateSpace
; CHECK: lbzux
}

!0 = metadata !{metadata !"any pointer", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
