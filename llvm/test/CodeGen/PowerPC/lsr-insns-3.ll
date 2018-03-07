; RUN: llc < %s -O2 -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s

; LLC checks that LSR prefers less instructions to less induction variables
; Without the PPC specific LSR cost model, extra addition instructions
; will occur within the loop before the call to _ZN6myTypeC1Ev.

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%struct.myType2 = type <{ i32, i8, %struct.myType, [2 x i8] }>
%struct.myType = type { i8 }

define nonnull %struct.myType2* @_Z6myIniti(i64 signext %n) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: _Z6myIniti:
; CHECK:  [[LABEL1:.LBB[0-9A-Z_]+]]:
; CHECK:    mr {{[0-9]+}}, [[REG1:[0-9]+]]
; CHECK-NEXT:    bl _ZN6myTypeC1Ev
; CHECK:    addi [[REG2:[0-9]+]], [[REG2]], -8
; CHECK-NEXT:    addi [[REG1]], [[REG1]], 8
; CHECK-NEXT:    cmpldi [[REG2]], 0
; CHECK-NEXT:    bne 0, [[LABEL1]]

entry:
  %call = tail call i8* @_Znam(i64 %n) #5
  %cast = bitcast i8* %call to %struct.myType2*
  %arrayctor.end = getelementptr inbounds %struct.myType2, %struct.myType2* %cast, i64 %n
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %invoke.cont, %new.ctorloop
  %arrayctor.cur = phi %struct.myType2* [ %cast, %entry ], [ %arrayctor.next, %invoke.cont ]
  %x.i = getelementptr inbounds %struct.myType2, %struct.myType2* %arrayctor.cur, i64 0, i32 2
  invoke void @_ZN6myTypeC1Ev(%struct.myType* nonnull %x.i)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %arrayctor.loop
  %arrayctor.next = getelementptr inbounds %struct.myType2, %struct.myType2* %arrayctor.cur, i64 1
  %arrayctor.done = icmp eq %struct.myType2* %arrayctor.next, %arrayctor.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %invoke.cont, %entry
  ret %struct.myType2* %cast

lpad:                                             ; preds = %arrayctor.loop
  %landing = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZdaPv(i8* nonnull %call) #6
  resume { i8*, i32 } %landing
}

declare noalias nonnull i8* @_Znam(i64) local_unnamed_addr #2

declare i32 @__gxx_personality_v0(...)

declare void @_ZdaPv(i8*) local_unnamed_addr #3

declare void @_ZN6myTypeC1Ev(%struct.myType*) unnamed_addr #4

