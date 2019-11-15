; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Check IndexedReference::computeRefCost can handle type differences between
; Stride and TripCount

; CHECK: Loop 'for.cond' has cost = 64

%struct._Handleitem = type { %struct._Handleitem* }

define void @handle_to_ptr(%struct._Handleitem** %blocks) {
; Preheader:
entry:
  br label %for.cond

; Loop:
for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 1, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds %struct._Handleitem*, %struct._Handleitem** %blocks, i64 %idxprom
  store %struct._Handleitem* null, %struct._Handleitem** %arrayidx, align 8
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

; Exit blocks
for.end:                                          ; preds = %for.cond
  ret void

}
