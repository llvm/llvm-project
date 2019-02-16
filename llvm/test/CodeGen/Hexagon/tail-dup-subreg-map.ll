; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s
; REQUIRES: asserts

; When tail-duplicating a block with PHI nodes that use subregisters, the
; subregisters were dropped by the tail duplicator, resulting in invalid
; COPY instructions being generated.

; CHECK: = asl(r{{[0-9]+}}:{{[0-9]+}},#15)

target triple = "hexagon"

%struct.0 = type { i64, i16 }
%struct.1 = type { i64, i64 }

declare hidden fastcc void @foo(%struct.0* noalias nocapture, i8 signext, i8 zeroext, i32, i64, i64) unnamed_addr #0

define void @fred(%struct.0* noalias nocapture sret %agg.result, %struct.1* byval nocapture readonly align 8 %a) #1 {
entry:
  %0 = load i64, i64* undef, align 8
  switch i32 undef, label %if.else [
    i32 32767, label %if.then
    i32 0, label %if.then7
  ]

if.then:                                          ; preds = %entry
  ret void

if.then7:                                         ; preds = %entry
  br i1 undef, label %if.then.i, label %if.else16.i

if.then.i:                                        ; preds = %if.then7
  br i1 undef, label %if.then5.i, label %if.else.i

if.then5.i:                                       ; preds = %if.then.i
  %shl.i21 = shl i64 %0, 0
  br label %if.end.i

if.else.i:                                        ; preds = %if.then.i
  %shl12.i = shl i64 %0, 7
  br label %if.end.i

if.end.i:                                         ; preds = %if.else.i, %if.then5.i
  %aSig0.0 = phi i64 [ undef, %if.then5.i ], [ %shl12.i, %if.else.i ]
  %storemerge43.i = phi i64 [ %shl.i21, %if.then5.i ], [ 0, %if.else.i ]
  %sub15.i = sub nsw i32 -63, 8
  br label %if.end13

if.else16.i:                                      ; preds = %if.then7
  br label %if.end13

if.else:                                          ; preds = %entry
  %or12 = or i64 9, 281474976710656
  br label %if.end13

if.end13:                                         ; preds = %if.else, %if.else16.i, %if.end.i
  %aSig1.1 = phi i64 [ %0, %if.else ], [ %storemerge43.i, %if.end.i ], [ undef, %if.else16.i ]
  %aSig0.2 = phi i64 [ %or12, %if.else ], [ %aSig0.0, %if.end.i ], [ undef, %if.else16.i ]
  %aExp.0 = phi i32 [ undef, %if.else ], [ %sub15.i, %if.end.i ], [ undef, %if.else16.i ]
  %shl2.i = shl i64 %aSig0.2, 15
  %shr.i = lshr i64 %aSig1.1, 49
  %or.i = or i64 %shl2.i, %shr.i
  tail call fastcc void @foo(%struct.0* noalias %agg.result, i8 signext 80, i8 zeroext undef, i32 %aExp.0, i64 %or.i, i64 undef)
  unreachable
}

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind }
