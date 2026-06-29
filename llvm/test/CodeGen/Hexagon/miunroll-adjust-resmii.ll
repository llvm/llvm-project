; RUN: llc -O3  -march=hexagon -enable-machine-unroller=false < %s | FileCheck --check-prefix=CHECK-NO-UNROLL %s
; RUN: llc -O3  -march=hexagon -enable-machine-unroller=true < %s | FileCheck --check-prefix=CHECK-UNROLL %s

; Without the machine unroller, make sure that the inner most loop has only one sfmpy instruction.
; CHECK-NO-UNROLL: loop0(.LBB0_[[LOOP:.]]
; CHECK-NO-UNROLL: .LBB0_[[LOOP]]:
; CHECK-NO-UNROLL: sfmpy
; CHECK-NO-UNROLL-NOT: sfmpy
; CHECK-NO-UNROLL: endloop0
; CHECK-NO-UNROLL-NOT: loop0

; When the machine unroller is enabled, the inner most loop in the test
; gets unrolled by 2. Make sure that there are 2 sfmpy instructions
; (one for each loop iteration) in the unrolled loop.

; CHECK-UNROLL: loop0(.LBB0_[[LOOP:.]]
; CHECK-UNROLL: .LBB0_[[LOOP]]:
; CHECK-UNROLL: sfmpy
; CHECK-UNROLL: sfmpy
; CHECK-UNROLL-NOT: sfmpy
; CHECK-UNROLL: } :endloop0

; Function Attrs: noinline nounwind
define dso_local void @test(ptr noalias nocapture readonly %in, ptr noalias nocapture %out, float %scale, i32 %n_samples) local_unnamed_addr {
entry:
  %cmp6 = icmp eq i32 %n_samples, 0
  br i1 %cmp6, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %arrayidx.phi = phi ptr [ %arrayidx.inc, %for.body ], [ %in, %entry ]
  %arrayidx1.phi = phi ptr [ %arrayidx1.inc, %for.body ], [ %out, %entry ]
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %0 = load i32, ptr %arrayidx.phi, align 4
  %1 = tail call float @llvm.hexagon.F2.conv.w2sf(i32 %0)
  %mul = fmul contract float %1, %scale
  store float %mul, ptr %arrayidx1.phi, align 4
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %n_samples
  %arrayidx.inc = getelementptr i32, ptr %arrayidx.phi, i32 1
  %arrayidx1.inc = getelementptr float, ptr %arrayidx1.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.hexagon.F2.conv.w2sf(i32)
