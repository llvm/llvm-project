; RUN: opt %loadPolly -polly-detect -analyze < %s \
; RUN:  | FileCheck %s -check-prefix=DETECT

; RUN: opt %loadPolly -polly-scops -analyze < %s \
; RUN:  | FileCheck %s -check-prefix=SCOPS

; DETECT: Valid Region for Scop: if.end116 => for.inc216
; SCOPS-NOT: Statements

; Verify that we detect this scop, but that, due to an infeasible run-time
; check, we refuse to model it.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.tnode.1.7.13.67.121.175.217.307.325.337.349.367.379.391.433.445.667.727.733.739 = type { i32, i32, %struct.tnode.1.7.13.67.121.175.217.307.325.337.349.367.379.391.433.445.667.727.733.739*, %struct.tnode.1.7.13.67.121.175.217.307.325.337.349.367.379.391.433.445.667.727.733.739*, %struct.tnode.1.7.13.67.121.175.217.307.325.337.349.367.379.391.433.445.667.727.733.739*, %struct.tnode.1.7.13.67.121.175.217.307.325.337.349.367.379.391.433.445.667.727.733.739*, %struct.tnode.1.7.13.67.121.175.217.307.325.337.349.367.379.391.433.445.667.727.733.739*, i32, i32, %struct.plist.0.6.12.66.120.174.216.306.324.336.348.366.378.390.432.444.666.726.732.738* }
%struct.plist.0.6.12.66.120.174.216.306.324.336.348.366.378.390.432.444.666.726.732.738 = type { i32, %struct.plist.0.6.12.66.120.174.216.306.324.336.348.366.378.390.432.444.666.726.732.738* }

@vFixedEdgeRoot = external global %struct.tnode.1.7.13.67.121.175.217.307.325.337.349.367.379.391.433.445.667.727.733.739*, align 8
@hEdgeRoot = external global %struct.tnode.1.7.13.67.121.175.217.307.325.337.349.367.379.391.433.445.667.727.733.739*, align 8

; Function Attrs: nounwind uwtable
define void @readgeo() #0 {
entry:
  %vx = alloca i32, align 4
  br label %if.end64

if.end64:                                         ; preds = %entry
  br label %for.body73

for.body73:                                       ; preds = %for.inc216, %if.end64
  %v.0101 = phi i32 [ 0, %for.inc216 ], [ 1, %if.end64 ]
  br i1 undef, label %if.then93, label %if.else

if.then93:                                        ; preds = %for.body73
  br label %for.inc216

if.else:                                          ; preds = %for.body73
  br i1 undef, label %if.then111, label %if.end116

if.then111:                                       ; preds = %if.else
  br label %if.end116

if.end116:                                        ; preds = %if.then111, %if.else
  %rippleCount.2 = phi i32 [ 1, %if.then111 ], [ undef, %if.else ]
  %rem11790 = and i32 %v.0101, 1
  %cmp118 = icmp eq i32 %rem11790, 0
  br i1 %cmp118, label %if.then120, label %if.else154

if.then120:                                       ; preds = %if.end116
  call void @tinsert()
  br label %if.end193

if.else154:                                       ; preds = %if.end116
  call void @tinsert()
  br label %if.end193

if.end193:                                        ; preds = %if.else154, %if.then120
  %0 = load i32, i32* %vx, align 4
  br label %for.inc216

for.inc216:                                       ; preds = %if.end193, %if.then93
  %rippleCount.3 = phi i32 [ undef, %if.then93 ], [ %rippleCount.2, %if.end193 ]
  %ux.2 = phi i32 [ undef, %if.then93 ], [ %0, %if.end193 ]
  br i1 undef, label %for.body73, label %for.end218

for.end218:                                       ; preds = %for.inc216
  unreachable
}

declare void @tinsert()
