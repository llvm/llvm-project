; RUN: llc -O3  -march=hexagon -enable-machine-unroller=false < %s | FileCheck --check-prefix=CHECK-NO-UNROLL %s
; RUN: llc -O3  -march=hexagon -enable-machine-unroller=true -enable-timing-class-latency=false -enable-bsb-sched=true < %s | FileCheck --check-prefix=CHECK-UNROLL %s

; correctness test for machine unroller

; bsb generates 1.5 packets per loop iteration and better hides load latency with 2x unrolling
; LBB0_8:
;	{
;		r12 = sfadd(r12,r9)
;		r9 = sfmpy(r6,r7)
;		r13 = memw(r3+#0)
;		r14 = memw(r5+#0)
;	}
;	{
;		r3 = add(r3,#256)
;		r5 = add(r5,#256)
;		r6 = memw(r3+#128)
;		r7 = memw(r5+#128)
;	}
;	{
;		r12 = sfadd(r12,r8)
;		r8 = sfmpy(r13,r14)
;
;	} :endloop0
; b2b generates 2 packets per loop iteration and does not unroll
;.LBB0_8:
;       {
;		r0 = sfadd(r0,r5)
;		r6 = sfmpy(r3,r4)
;		r5 = r6
;		r3 = memw(r7+#0)
;	}
;	{
;		r7 = add(r7,#128)
;		r8 = add(r8,#128)
;		r4 = memw(r8+#0)
;	} :endloop0
; create b2b bug


; Without the machine unroller, make sure that the inner most loop has only one sfmpy instruction.

; CHECK-NO-UNROLL: loop0(.LBB0_[[LOOP:.*]],
; CHECK-NO-UNROLL: if ({{.*}}p{{[0-3]}}) jump{{.*}} .LBB0_{{.*}}
; CHECK-NO-UNROLL: .LBB0_[[LOOP]]:
; CHECK-NO-UNROLL-DAG: {
; CHECK-NO-UNROLL-DAG: sfmpy
; CHECK-NO-UNROLL-NOT: sfmpy
; CHECK-NO-UNROLL: endloop0
; CHECK-NO-UNROLL-NOT: loop0

; When the machine unroller is enabled, the inner most loop in the test
; gets unrolled by 2. Make sure that there are only 3 packets and
; 2 sfmpy instructions (one for each loop iteration) in the unrolled loop.

; CHECK-UNROLL: loop0(.LBB0_[[LOOP:.]]
; CHECK-UNROLL: .LBB0_[[LOOP]]:
; CHECK-UNROLL: sfmpy
; CHECK-UNROLL: sfmpy
; CHECK-UNROLL-NOT: sfmpy
; CHECK-UNROLL: } :endloop0

%struct.loops_params_s = type { i32, i32, i32, i32, i32, i32, i32, [32 x i32], [32 x i32], i32, i32, i32, i32, i32, ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, float, ptr, ptr, i32 }

define float @inner_product(ptr %p) {
entry:
  %v = getelementptr inbounds %struct.loops_params_s, ptr %p, i32 0, i32 17
  %0 = load ptr, ptr %v, align 4
  %1 = load ptr, ptr %0, align 4
  %arrayidx2 = getelementptr inbounds ptr, ptr %0, i32 1
  %2 = load ptr, ptr %arrayidx2, align 4
  %N = getelementptr inbounds %struct.loops_params_s, ptr %p, i32 0, i32 5
  %3 = load i32, ptr %N, align 4
  %Loop = getelementptr inbounds %struct.loops_params_s, ptr %p, i32 0, i32 9
  %4 = load i32, ptr %Loop, align 4
  %vsize = getelementptr inbounds %struct.loops_params_s, ptr %p, i32 0, i32 1
  %5 = load i32, ptr %vsize, align 4
  %call = tail call i32 @reinit_vec(ptr %p, ptr %1, i32 %5)
  %6 = load i32, ptr %vsize, align 4
  %call4 = tail call i32 @reinit_vec(ptr %p, ptr %2, i32 %6)
  %cmp39 = icmp slt i32 %4, 1
  br i1 %cmp39, label %for.end13, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %cmp636 = icmp sgt i32 %3, 0
  br label %for.body

for.body:                                         ; preds = %for.inc11, %for.body.lr.ph
  %q.042 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %q.1.lcssa, %for.inc11 ]
  %l.040 = phi i32 [ 1, %for.body.lr.ph ], [ %inc12, %for.inc11 ]
  br i1 %cmp636, label %for.body7.lr.ph, label %for.inc11

for.body7.lr.ph:                                  ; preds = %for.body
  %arrayidx8.gep = getelementptr float, ptr %2, i32 %l.040
  br label %for.body7

for.body7:                                        ; preds = %for.body7, %for.body7.lr.ph
  %q.138 = phi float [ %q.042, %for.body7.lr.ph ], [ %add10, %for.body7 ]
  %arrayidx8.phi = phi ptr [ %arrayidx8.gep, %for.body7.lr.ph ], [ %arrayidx8.inc, %for.body7 ]
  %arrayidx9.phi = phi ptr [ %1, %for.body7.lr.ph ], [ %arrayidx9.inc, %for.body7 ]
  %k.037 = phi i32 [ 0, %for.body7.lr.ph ], [ %inc, %for.body7 ]
  %7 = load float, ptr %arrayidx8.phi, align 4
  %8 = load float, ptr %arrayidx9.phi, align 4
  %mul = fmul float %7, %8
  %add10 = fadd float %q.138, %mul
  %inc = add nuw nsw i32 %k.037, 1
  %exitcond = icmp eq i32 %inc, %3
  %arrayidx8.inc = getelementptr float, ptr %arrayidx8.phi, i32 32
  %arrayidx9.inc = getelementptr float, ptr %arrayidx9.phi, i32 32
  br i1 %exitcond, label %for.inc11, label %for.body7

for.inc11:                                        ; preds = %for.body7, %for.body
  %q.1.lcssa = phi float [ %q.042, %for.body ], [ %add10, %for.body7 ]
  %inc12 = add nuw nsw i32 %l.040, 1
  %exitcond44 = icmp eq i32 %l.040, %4
  br i1 %exitcond44, label %for.end13, label %for.body

for.end13:                                        ; preds = %for.inc11, %entry
  %q.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %q.1.lcssa, %for.inc11 ]
  ret float %q.0.lcssa
}

declare i32 @reinit_vec(...) local_unnamed_addr
