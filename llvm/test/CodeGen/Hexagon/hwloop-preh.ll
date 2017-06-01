; RUN: llc -march=hexagon -disable-machine-licm -hwloop-spec-preheader=1 < %s | FileCheck %s
; CHECK: loop0

target triple = "hexagon"

define i32 @foo(i32 %x, i32 %n, i32* nocapture %A, i32* nocapture %B) #0 {
entry:
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %for.cond.preheader, label %return

for.cond.preheader:                               ; preds = %entry
  %cmp16 = icmp sgt i32 %n, 0
  br i1 %cmp16, label %for.body.preheader, label %return

for.body.preheader:                               ; preds = %for.cond.preheader
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %arrayidx.phi = phi i32* [ %arrayidx.inc, %for.body ], [ %B, %for.body.preheader ]
  %arrayidx2.phi = phi i32* [ %arrayidx2.inc, %for.body ], [ %A, %for.body.preheader ]
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %0 = load i32, i32* %arrayidx.phi, align 4, !tbaa !0
  %1 = load i32, i32* %arrayidx2.phi, align 4, !tbaa !0
  %add = add nsw i32 %1, %0
  store i32 %add, i32* %arrayidx2.phi, align 4, !tbaa !0
  %inc = add nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %n
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  %arrayidx2.inc = getelementptr i32, i32* %arrayidx2.phi, i32 1
  br i1 %exitcond, label %return.loopexit, label %for.body

return.loopexit:                                  ; preds = %for.body
  br label %return

return:                                           ; preds = %return.loopexit, %for.cond.preheader, %entry
  %retval.0 = phi i32 [ 2, %entry ], [ 0, %for.cond.preheader ], [ 0, %return.loopexit ]
  ret i32 %retval.0
}

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="-hvx,-hvx-double" }
