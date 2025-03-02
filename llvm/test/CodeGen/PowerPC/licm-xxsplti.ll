;; Test hoisting `xxspltib` out of the loop.

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff --mcpu=pwr10 \
; RUN:   %s -o - 2>&1 | FileCheck --check-prefix=AIX64 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff --mcpu=pwr10 \
; RUN:   %s -o - 2>&1 | FileCheck --check-prefix=AIX32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64le-unknown-linux-gnu --mcpu=pwr10 \
; RUN:   %s -o - 2>&1 | FileCheck --check-prefix=LINUX64LE %s

define void @_Z3fooPfS_Pi(ptr noalias nocapture noundef %_a, ptr noalias nocapture %In_a, ptr noalias nocapture %n) {
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

; AIX32:      ._Z3fooPfS_Pi:
; AIX32-NEXT: # %bb.0:                                # %entry
; AIX32-NEXT:   lwz 5, 0(5)
; AIX32-NEXT:   cmpwi   5, 1
; AIX32-NEXT:   bltlr   0
; AIX32-NEXT: # %bb.1:                                # %for.body.preheader
; AIX32-NEXT:   li 6, 0
; AIX32-NEXT:   beq     0, L..BB0_4
; AIX32-NEXT: # %bb.2:                                # %for.body.preheader.new
; AIX32-NEXT:   addi 12, 4, -8
; AIX32-NEXT:   addi 9, 3, -8
; AIX32-NEXT:   rlwinm 7, 5, 0, 1, 30
; AIX32-NEXT:   li 8, 0
; AIX32-NEXT:   li 10, 8
; AIX32-NEXT:   li 11, 12
; AIX32-NEXT:   .align  4
; AIX32-NEXT: L..BB0_3:                               # %for.body
; AIX32-NEXT:                                         # =>This Inner Loop Header: Depth=1
; AIX32-NEXT:   lxvwsx 0, 12, 10
; AIX32-NEXT:   xxspltib 1, 6
; AIX32-NEXT:   lxvwsx 2, 12, 11
; AIX32-NEXT:   addic 6, 6, 2
; AIX32-NEXT:   addi 12, 12, 8
; AIX32-NEXT:   addze 8, 8
; AIX32-NEXT:   xor 0, 6, 7
; AIX32-NEXT:   or. 0, 0, 8
; AIX32-NEXT:   xxland 0, 0, 1
; AIX32-NEXT:   xxland 1, 2, 1
; AIX32-NEXT:   xscvspdpn 0, 0
; AIX32-NEXT:   stfsu 0, 8(9)
; AIX32-NEXT:   xscvspdpn 0, 1
; AIX32-NEXT:   stfs 0, 4(9)
; AIX32-NEXT:   bne     0, L..BB0_3

; AIX64:      ._Z3fooPfS_Pi:
; AIX64-NEXT: # %bb.0:                                # %entry
; AIX64-NEXT:   lwz 5, 0(5)
; AIX64-NEXT:   cmpwi   5, 1
; AIX64-NEXT:   bltlr   0
; AIX64-NEXT: # %bb.1:                                # %for.body.preheader
; AIX64-NEXT:   li 6, 0
; AIX64-NEXT:   cmplwi  5, 1
; AIX64-NEXT:   beq     0, L..BB0_4
; AIX64-NEXT: # %bb.2:                                # %for.body.preheader.new
; AIX64-NEXT:   rlwinm 6, 5, 0, 1, 30
; AIX64-NEXT:   addi 10, 4, -8
; AIX64-NEXT:   addi 7, 3, -8
; AIX64-NEXT:   li 8, 8
; AIX64-NEXT:   li 9, 12
; AIX64-NEXT:   li 11, 4
; AIX64-NEXT:   addi 6, 6, -2
; AIX64-NEXT:   rldicl 6, 6, 63, 1
; AIX64-NEXT:   addi 6, 6, 1
; AIX64-NEXT:   mtctr 6
; AIX64-NEXT:   li 6, 0
; AIX64-NEXT:   .align  4
; AIX64-NEXT: L..BB0_3:                               # %for.body
; AIX64-NEXT:                                         # =>This Inner Loop Header: Depth=1
; AIX64-NEXT:   lxvwsx 0, 10, 8
; AIX64-NEXT:   xxspltib 1, 6
; AIX64-NEXT:   addi 6, 6, 2
; AIX64-NEXT:   xxland 0, 0, 1
; AIX64-NEXT:   xscvspdpn 0, 0
; AIX64-NEXT:   stfsu 0, 8(7)
; AIX64-NEXT:   lxvwsx 0, 10, 9
; AIX64-NEXT:   addi 10, 10, 8
; AIX64-NEXT:   xxland 0, 0, 1
; AIX64-NEXT:   xxsldwi 0, 0, 0, 3
; AIX64-NEXT:   stfiwx 0, 7, 11
; AIX64-NEXT:   bdnz L..BB0_3

; LINUX64LE:      _Z3fooPfS_Pi:                           # @_Z3fooPfS_Pi
; LINUX64LE-NEXT: .Lfunc_begin0:
; LINUX64LE-NEXT:       .cfi_startproc
; LINUX64LE-NEXT: # %bb.0:                                # %entry
; LINUX64LE-NEXT:       lwz 5, 0(5)
; LINUX64LE-NEXT:       cmpwi   5, 1
; LINUX64LE-NEXT:       bltlr   0
; LINUX64LE-NEXT: # %bb.1:                                # %for.body.preheader
; LINUX64LE-NEXT:       li 6, 0
; LINUX64LE-NEXT:       cmplwi  5, 1
; LINUX64LE-NEXT:       beq     0, .LBB0_4
; LINUX64LE-NEXT: # %bb.2:                                # %for.body.preheader.new
; LINUX64LE-NEXT:       rlwinm 6, 5, 0, 1, 30
; LINUX64LE-NEXT:       addi 8, 4, -8
; LINUX64LE-NEXT:       addi 7, 3, -8
; LINUX64LE-NEXT:       li 9, 8
; LINUX64LE-NEXT:       li 10, 12
; LINUX64LE-NEXT:       li 11, 4
; LINUX64LE-NEXT:       addi 6, 6, -2
; LINUX64LE-NEXT:       rldicl 6, 6, 63, 1
; LINUX64LE-NEXT:       addi 6, 6, 1
; LINUX64LE-NEXT:       mtctr 6
; LINUX64LE-NEXT:       li 6, 0
; LINUX64LE-NEXT:       .p2align        4
; LINUX64LE-NEXT: .LBB0_3:                                # %for.body
; LINUX64LE-NEXT:                                         # =>This Inner Loop Header: Depth=1
; LINUX64LE-NEXT:       lxvwsx 0, 8, 9
; LINUX64LE-NEXT:       xxspltib 1, 6
; LINUX64LE-NEXT:       addi 6, 6, 2
; LINUX64LE-NEXT:       xxland 0, 0, 1
; LINUX64LE-NEXT:       xxsldwi 0, 0, 0, 3
; LINUX64LE-NEXT:       xscvspdpn 0, 0
; LINUX64LE-NEXT:       stfsu 0, 8(7)
; LINUX64LE-NEXT:       lxvwsx 0, 8, 10
; LINUX64LE-NEXT:       addi 8, 8, 8
; LINUX64LE-NEXT:       xxland 0, 0, 1
; LINUX64LE-NEXT:       stxvrwx 0, 7, 11
; LINUX64LE-NEXT:       bdnz .LBB0_3
