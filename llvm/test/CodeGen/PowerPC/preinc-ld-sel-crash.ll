; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

%t1 = type { ptr, ptr }
%t2 = type <{ ptr, i32, [4 x i8] }>
%t3 = type { ptr }

@_ZN4Foam10SLListBase13endConstIter_E = external global %t1

define void @_ZN4FoamrsIbEERNS_7IstreamES2_RNS_4ListIT_EE() #0 {
entry:
  switch i32 undef, label %if.else82 [
    i32 9, label %if.then
    i32 6, label %invoke.cont10
    i32 1, label %invoke.cont61
  ]

if.then:                                          ; preds = %entry
  unreachable

invoke.cont10:                                    ; preds = %entry
  unreachable

invoke.cont61:                                    ; preds = %entry
  br i1 undef, label %if.end75, label %if.then64

if.then64:                                        ; preds = %invoke.cont61
  unreachable

if.end75:                                         ; preds = %invoke.cont61
  br i1 undef, label %if.then17.i, label %if.then.i181

if.then.i181:                                     ; preds = %if.end75
  unreachable

if.then17.i:                                      ; preds = %if.end75
  %tobool.i.i.i = icmp eq i32 undef, 0
  %0 = load ptr, ptr undef, align 8
  %agg.tmp.sroa.3.0.copyload33.in.i = select i1 %tobool.i.i.i, ptr getelementptr inbounds (%t1, ptr @_ZN4Foam10SLListBase13endConstIter_E, i64 0, i32 1), ptr %0
  %agg.tmp.sroa.3.0.copyload33.i = load i64, ptr %agg.tmp.sroa.3.0.copyload33.in.i, align 8
  %1 = inttoptr i64 %agg.tmp.sroa.3.0.copyload33.i to ptr
  %2 = load ptr, ptr getelementptr inbounds (%t1, ptr @_ZN4Foam10SLListBase13endConstIter_E, i64 0, i32 1), align 8
  %cmp.i37.i = icmp eq ptr %1, %2
  br i1 %cmp.i37.i, label %invoke.cont79, label %for.body.lr.ph.i

; CHECK-LABEL: @_ZN4FoamrsIbEERNS_7IstreamES2_RNS_4ListIT_EE

for.body.lr.ph.i:                                 ; preds = %if.then17.i
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.lr.ph.i
  br i1 undef, label %invoke.cont79, label %for.body.i

invoke.cont79:                                    ; preds = %for.body.i, %if.then17.i
  unreachable

if.else82:                                        ; preds = %entry
  ret void
}

attributes #0 = { "target-cpu"="a2q" }

