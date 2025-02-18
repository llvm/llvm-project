;; Test hoisting `xxspltib` out the loop.

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff  \
; RUN:   %s -o - 2>&1 | FileCheck  %s

define void @_Z3fooPfS_Pi(ptr noalias nocapture noundef writeonly %_a, ptr noalias nocapture noundef readonly %In_a, ptr noalias nocapture noundef readonly %n) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr %n, align 4
  %cmp9 = icmp sgt i32 %0, 0
  br i1 %cmp9, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               
  %wide.trip.count = zext nneg i32 %0 to i64
  %xtraiter = and i64 %wide.trip.count, 1
  %1 = icmp eq i32 %0, 1
  br i1 %1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           
  %unroll_iter = and i64 %wide.trip.count, 2147483646
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              
  %indvars.iv.unr = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next.1, %for.body ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    
  %arrayidx.epil = getelementptr inbounds nuw float, ptr %In_a, i64 %indvars.iv.unr
  %2 = load float, ptr %arrayidx.epil, align 4
  %vecins.i.epil = insertelement <4 x float> poison, float %2, i64 0
  %3 = bitcast <4 x float> %vecins.i.epil to <16 x i8>
  %and1.i.epil = and <16 x i8> %3, <i8 6, i8 6, i8 6, i8 6, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %4 = bitcast <16 x i8> %and1.i.epil to <4 x float>
  %vecext.i.epil = extractelement <4 x float> %4, i64 0
  %arrayidx5.epil = getelementptr inbounds nuw float, ptr %_a, i64 %indvars.iv.unr
  store float %vecext.i.epil, ptr %arrayidx5.epil, align 4
  br label %for.cond.cleanup

for.cond.cleanup:                                 
  ret void

for.body:                                         
  %indvars.iv = phi i64 [ 0, %for.body.preheader.new ], [ %indvars.iv.next.1, %for.body ]
  %niter = phi i64 [ 0, %for.body.preheader.new ], [ %niter.next.1, %for.body ]
  %arrayidx = getelementptr inbounds nuw float, ptr %In_a, i64 %indvars.iv
  %5 = load float, ptr %arrayidx, align 4
  %vecins.i = insertelement <4 x float> poison, float %5, i64 0
  %6 = bitcast <4 x float> %vecins.i to <16 x i8>
  %and1.i = and <16 x i8> %6, <i8 6, i8 6, i8 6, i8 6, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %7 = bitcast <16 x i8> %and1.i to <4 x float>
  %vecext.i = extractelement <4 x float> %7, i64 0
  %arrayidx5 = getelementptr inbounds nuw float, ptr %_a, i64 %indvars.iv
  store float %vecext.i, ptr %arrayidx5, align 4
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %arrayidx.1 = getelementptr inbounds nuw float, ptr %In_a, i64 %indvars.iv.next
  %8 = load float, ptr %arrayidx.1, align 4
  %vecins.i.1 = insertelement <4 x float> poison, float %8, i64 0
  %9 = bitcast <4 x float> %vecins.i.1 to <16 x i8>
  %and1.i.1 = and <16 x i8> %9, <i8 6, i8 6, i8 6, i8 6, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %10 = bitcast <16 x i8> %and1.i.1 to <4 x float>
  %vecext.i.1 = extractelement <4 x float> %10, i64 0
  %arrayidx5.1 = getelementptr inbounds nuw float, ptr %_a, i64 %indvars.iv.next
  store float %vecext.i.1, ptr %arrayidx5.1, align 4
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2
  %niter.next.1 = add i64 %niter, 2
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter
  br i1 %niter.ncmp.1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr10" "target-features"="+altivec,+bpermd,+crbits,+crypto,+direct-move,+extdiv,+isa-v206-instructions,+isa-v207-instructions,+isa-v30-instructions,+isa-v31-instructions,+mma,+paired-vector-memops,+pcrelative-memops,+power10-vector,+power8-vector,+power9-vector,+prefix-instrs,+quadword-atomics,+vsx,-aix-shared-lib-tls-model-opt,-aix-small-local-dynamic-tls,-aix-small-local-exec-tls,-htm,-privileged,-rop-protect,-spe" }

; CHECK:      ._Z3fooPfS_Pi:
; CHECK-NEXT: # %bb.0:                                # %entry
; CHECK-NEXT:   lwz 5, 0(5)
; CHECK-NEXT:   cmpwi   5, 1
; CHECK-NEXT:   bltlr   0
; CHECK-NEXT: # %bb.1:                                # %for.body.preheader
; CHECK-NEXT:   li 6, 0
; CHECK-NEXT:   cmplwi  5, 1
; CHECK-NEXT:   beq     0, L..BB0_4
; CHECK-NEXT: # %bb.2:                                # %for.body.preheader.new
; CHECK-NEXT:   rlwinm 6, 5, 0, 1, 30
; CHECK-NEXT:   addi 10, 4, -8
; CHECK-NEXT:   addi 7, 3, -8
; CHECK-NEXT:   li 8, 8
; CHECK-NEXT:   li 9, 12
; CHECK-NEXT:   li 11, 4
; CHECK-NEXT:   addi 6, 6, -2
; CHECK-NEXT:   rldicl 6, 6, 63, 1
; CHECK-NEXT:   addi 6, 6, 1
; CHECK-NEXT:   mtctr 6
; CHECK-NEXT:   li 6, 0
; CHECK-NEXT:   .align  4
; CHECK-NEXT: L..BB0_3:                               # %for.body
; CHECK-NEXT:                                         # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:   lxvwsx 0, 10, 8
; CHECK-NEXT:   xxspltib 1, 6
; CHECK-NEXT:   addi 6, 6, 2
; CHECK-NEXT:   xxland 0, 0, 1
; CHECK-NEXT:   xscvspdpn 0, 0
; CHECK-NEXT:   stfsu 0, 8(7)
; CHECK-NEXT:   lxvwsx 0, 10, 9
; CHECK-NEXT:   addi 10, 10, 8
; CHECK-NEXT:   xxland 0, 0, 1
; CHECK-NEXT:   xxsldwi 0, 0, 0, 3
; CHECK-NEXT:   stfiwx 0, 7, 11
; CHECK-NEXT:   bdnz L..BB0_3
