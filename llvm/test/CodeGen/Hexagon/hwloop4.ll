; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
;
; Remove the unnecessary 'add' instruction used for the hardware loop setup.

; CHECK: [[OP0:r[0-9]+]] = add([[OP1:r[0-9]+]],#-[[OP2:[0-9]+]]
; CHECK-NOT: add([[OP0]],#[[OP2]])
; CHECK: lsr([[OP1]],#{{[0-9]+}})
; CHECK: loop0

define void @matrix_mul_matrix(i32 %N, ptr nocapture %C, ptr nocapture readnone %A, ptr nocapture readnone %B) #0 {
entry:
  %cmp4 = icmp eq i32 %N, 0
  br i1 %cmp4, label %for.end, label %for.body.preheader

for.body.preheader:
  %maxval = add i32 %N, -7
  %0 = icmp sgt i32 %maxval, 0
  br i1 %0, label %for.body.preheader9, label %for.body.ur.preheader

for.body.preheader9:
  br label %for.body

for.body:
  %arrayidx.phi = phi ptr [ %arrayidx.inc.7, %for.body ], [ %C, %for.body.preheader9 ]
  %i.05 = phi i32 [ %inc.7, %for.body ], [ 0, %for.body.preheader9 ]
  store i32 %i.05, ptr %arrayidx.phi, align 4
  %inc = add i32 %i.05, 1
  %arrayidx.inc = getelementptr i32, ptr %arrayidx.phi, i32 1
  store i32 %inc, ptr %arrayidx.inc, align 4
  %inc.1 = add i32 %i.05, 2
  %arrayidx.inc.1 = getelementptr i32, ptr %arrayidx.phi, i32 2
  store i32 %inc.1, ptr %arrayidx.inc.1, align 4
  %inc.2 = add i32 %i.05, 3
  %arrayidx.inc.2 = getelementptr i32, ptr %arrayidx.phi, i32 3
  store i32 %inc.2, ptr %arrayidx.inc.2, align 4
  %inc.3 = add i32 %i.05, 4
  %arrayidx.inc.3 = getelementptr i32, ptr %arrayidx.phi, i32 4
  store i32 %inc.3, ptr %arrayidx.inc.3, align 4
  %inc.4 = add i32 %i.05, 5
  %arrayidx.inc.4 = getelementptr i32, ptr %arrayidx.phi, i32 5
  store i32 %inc.4, ptr %arrayidx.inc.4, align 4
  %inc.5 = add i32 %i.05, 6
  %arrayidx.inc.5 = getelementptr i32, ptr %arrayidx.phi, i32 6
  store i32 %inc.5, ptr %arrayidx.inc.5, align 4
  %inc.6 = add i32 %i.05, 7
  %arrayidx.inc.6 = getelementptr i32, ptr %arrayidx.phi, i32 7
  store i32 %inc.6, ptr %arrayidx.inc.6, align 4
  %inc.7 = add i32 %i.05, 8
  %exitcond.7 = icmp slt i32 %inc.7, %maxval
  %arrayidx.inc.7 = getelementptr i32, ptr %arrayidx.phi, i32 8
  br i1 %exitcond.7, label %for.body, label %for.end.loopexit.ur-lcssa

for.end.loopexit.ur-lcssa:
  %1 = icmp eq i32 %inc.7, %N
  br i1 %1, label %for.end, label %for.body.ur.preheader

for.body.ur.preheader:
  %arrayidx.phi.ur.ph = phi ptr [ %C, %for.body.preheader ], [ %arrayidx.inc.7, %for.end.loopexit.ur-lcssa ]
  %i.05.ur.ph = phi i32 [ 0, %for.body.preheader ], [ %inc.7, %for.end.loopexit.ur-lcssa ]
  br label %for.body.ur

for.body.ur:
  %arrayidx.phi.ur = phi ptr [ %arrayidx.inc.ur, %for.body.ur ], [ %arrayidx.phi.ur.ph, %for.body.ur.preheader ]
  %i.05.ur = phi i32 [ %inc.ur, %for.body.ur ], [ %i.05.ur.ph, %for.body.ur.preheader ]
  store i32 %i.05.ur, ptr %arrayidx.phi.ur, align 4
  %inc.ur = add i32 %i.05.ur, 1
  %exitcond.ur = icmp eq i32 %inc.ur, %N
  %arrayidx.inc.ur = getelementptr i32, ptr %arrayidx.phi.ur, i32 1
  br i1 %exitcond.ur, label %for.end.loopexit, label %for.body.ur

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
