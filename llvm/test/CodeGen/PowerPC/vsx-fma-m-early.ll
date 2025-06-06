;; Tests that the ppc-vsx-fma-mutate pass with the schedule-ppc-vsx-fma-mutation-early pass does not hoist xxspltiw out of loops.
; RUN: llc -verify-machineinstrs -mcpu=pwr10 -disable-ppc-vsx-fma-mutation=false \
; RUN:   -ppc-asm-full-reg-names -schedule-ppc-vsx-fma-mutation-early \
; RUN:    -mtriple powerpc64-ibm-aix < %s | FileCheck --check-prefixes=CHECK64,AIX64 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr10 -disable-ppc-vsx-fma-mutation=false \
; RUN:   -ppc-asm-full-reg-names -schedule-ppc-vsx-fma-mutation-early \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck --check-prefixes=CHECK64,LINUX64 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr10 -disable-ppc-vsx-fma-mutation=false \
; RUN:   -ppc-asm-full-reg-names -schedule-ppc-vsx-fma-mutation-early \
; RUN:    -mtriple powerpc-ibm-aix < %s | FileCheck --check-prefix=CHECK32 %s

define void @bar(ptr noalias nocapture noundef writeonly %__output_a, ptr noalias nocapture noundef readonly %var1321In_a, ptr noalias nocapture noundef readonly %n) {
entry:
  %0 = load i32, ptr %n, align 4
  %cmp11 = icmp sgt i32 %0, 0
  br i1 %cmp11, label %for.body.preheader, label %for.end

for.body.preheader:
  %wide.trip.count = zext i32 %0 to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %1 = shl nsw i64 %indvars.iv, 2
  %add.ptr = getelementptr inbounds float, ptr %var1321In_a, i64 %1
  %add.ptr.val = load <4 x float>, ptr %add.ptr, align 1
  %2 = tail call contract <4 x float> @llvm.fma.v4f32(<4 x float> %add.ptr.val, <4 x float> <float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000>, <4 x float> <float 6.270500e+03, float 6.270500e+03, float 6.270500e+03, float 6.270500e+03>)
  %add.ptr6 = getelementptr inbounds float, ptr %__output_a, i64 %1
  store <4 x float> %2, ptr %add.ptr6, align 1 
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret void
}

define void @foo(i1 %cmp97) #0 {
entry:
  br i1 %cmp97, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %0 = phi float [ %vecext.i, %for.body ], [ 0.000000e+00, %entry ]
  %splat.splatinsert.i = insertelement <4 x float> zeroinitializer, float %0, i64 0
  %1 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %splat.splatinsert.i, <4 x float> zeroinitializer, <4 x float> splat (float 6.270500e+03))
  %2 = tail call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> zeroinitializer, <4 x float> %splat.splatinsert.i)
  %3 = bitcast <4 x float> %1 to <4 x i32>
  %and1.i8896 = and <4 x i32> %2, %3
  %4 = bitcast <4 x i32> %and1.i8896 to <4 x float>
  %vecext.i = extractelement <4 x float> %4, i64 0
  br label %for.body

for.end:                                          ; preds = %entry
    ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) 

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float>, <4 x float>)

; CHECK64:      bar:
; CHECK64:      # %bb.0:                                # %entry
; CHECK64-NEXT:         lwz r5, 0(r5)
; CHECK64-NEXT:         cmpwi   r5, 1
; CHECK64-NEXT:         bltlr   cr0
; CHECK64-NEXT: # %bb.1:                                # %for.body.preheader
; CHECK64-NEXT:         xxspltiw vs0, 1069066811
; CHECK64-NEXT:         xxspltiw vs1, 1170469888
; CHECK64-NEXT:         mtctr r5
; CHECK64-NEXT:         li r5, 0
; CHECK64-NEXT:         {{.*}}align  5
; CHECK64-NEXT: [[L2_bar:.*]]:                               # %for.body
; CHECK64-NEXT:                                         # =>This Inner Loop Header: Depth=1
; CHECK64-NEXT:         lxvx vs2, r4, r5
; CHECK64-NEXT:         xvmaddmsp vs2, vs0, vs1
; CHECK64-NEXT:         stxvx vs2, r3, r5
; CHECK64-NEXT:         addi r5, r5, 16
; CHECK64-NEXT:         bdnz [[L2_bar]]
; CHECK64-NEXT: # %bb.3:                                # %for.end
; CHECK64-NEXT:         blr

; AIX64:      .foo:
; AIX64-NEXT: # %bb.0:                                # %entry
; AIX64-NEXT:   andi. r3, r3, 1
; AIX64-NEXT:   bclr 4, gt, 0
; AIX64-NEXT: # %bb.1:                                # %for.body.preheader
; AIX64-NEXT:   xxlxor f0, f0, f0
; AIX64-NEXT:   xxlxor vs1, vs1, vs1
; AIX64-NEXT:   xxlxor f2, f2, f2
; AIX64-NEXT:   .align  4
; AIX64-NEXT: L..BB1_2:                               # %for.body
; AIX64-NEXT:                                         # =>This Inner Loop Header: Depth=1
; AIX64-NEXT:   xxmrghd vs2, vs2, vs0
; AIX64-NEXT:   xvcvdpsp vs34, vs2
; AIX64-NEXT:   xxmrghd vs2, vs0, vs0
; AIX64-NEXT:   xvcvdpsp vs35, vs2
; AIX64-NEXT:   xxspltiw vs2, 1170469888
; AIX64-NEXT:   vmrgew v2, v2, v3
; AIX64-NEXT:   xvcmpgtsp vs3, vs1, vs34
; AIX64-NEXT:   xvmaddasp vs2, vs34, vs1
; AIX64-NEXT:   xxland vs2, vs3, vs2
; AIX64-NEXT:   xscvspdpn f2, vs2
; AIX64-NEXT:   b L..BB1_2

; LINUX64:      foo:                                    # @foo
; LINUX64-NEXT: .Lfunc_begin1:
; LINUX64-NEXT:         .cfi_startproc
; LINUX64-NEXT: # %bb.0:                                # %entry
; LINUX64-NEXT:         andi. r3, r3, 1
; LINUX64-NEXT:         bclr 4, gt, 0
; LINUX64-NEXT: # %bb.1:                                # %for.body.preheader
; LINUX64-NEXT:         xxlxor f0, f0, f0
; LINUX64-NEXT:         xxlxor vs1, vs1, vs1
; LINUX64-NEXT:         xxlxor f2, f2, f2
; LINUX64-NEXT:         .p2align        4
; LINUX64-NEXT: .LBB1_2:                                # %for.body
; LINUX64-NEXT:                                         # =>This Inner Loop Header: Depth=1
; LINUX64-NEXT:         xxmrghd vs2, vs0, vs2
; LINUX64-NEXT:         xvcvdpsp vs34, vs2
; LINUX64-NEXT:         xxspltd vs2, vs0, 0
; LINUX64-NEXT:         xvcvdpsp vs35, vs2
; LINUX64-NEXT:         xxspltiw vs2, 1170469888
; LINUX64-NEXT:         vmrgew v2, v3, v2
; LINUX64-NEXT:         xvcmpgtsp vs3, vs1, vs34
; LINUX64-NEXT:         xvmaddasp vs2, vs34, vs1
; LINUX64-NEXT:         xxland vs2, vs3, vs2
; LINUX64-NEXT:         xxsldwi vs2, vs2, vs2, 3
; LINUX64-NEXT:         xscvspdpn f2, vs2
; LINUX64-NEXT:         b .LBB1_2

; CHECK32:        .bar:
; CHECK32-NEXT: # %bb.0:                                # %entry
; CHECK32-NEXT:       lwz r5, 0(r5)
; CHECK32-NEXT:       cmpwi   r5, 0
; CHECK32-NEXT:       blelr cr0
; CHECK32-NEXT: # %bb.1:                                # %for.body.preheader
; CHECK32-NEXT:       xxspltiw vs0, 1069066811
; CHECK32-NEXT:       xxspltiw vs1, 1170469888
; CHECK32-NEXT:       li r6, 0
; CHECK32-NEXT:       li r7, 0
; CHECK32-NEXT:       .align  4
; CHECK32-NEXT: [[L2_foo:.*]]:                               # %for.body
; CHECK32-NEXT:                                         # =>This Inner Loop Header: Depth=1
; CHECK32-NEXT:       slwi r8, r7, 4
; CHECK32-NEXT:       addic r7, r7, 1
; CHECK32-NEXT:       addze r6, r6
; CHECK32-NEXT:       lxvx vs2, r4, r8
; CHECK32-NEXT:       xvmaddmsp vs2, vs0, vs1
; CHECK32-NEXT:       stxvx vs2, r3, r8
; CHECK32-NEXT:       xor r8, r7, r5
; CHECK32-NEXT:       or. r8, r8, r6
; CHECK32-NEXT:       bne     cr0, [[L2_foo]]

; CHECK32:      .foo:
; CHECK32-NEXT: # %bb.0:                                # %entry
; CHECK32-NEXT:         andi. r3, r3, 1
; CHECK32-NEXT:         bclr 4, gt, 0
; CHECK32-NEXT: # %bb.1:                                # %for.body.preheader
; CHECK32-NEXT:         lwz r3, L..C0(r2)                       # %const.0
; CHECK32-NEXT:         xxlxor f1, f1, f1
; CHECK32-NEXT:         xxlxor vs0, vs0, vs0
; CHECK32-NEXT:         xscvdpspn vs35, f1
; CHECK32-NEXT:         lxv vs34, 0(r3)
; CHECK32-NEXT:         .align  4
; CHECK32-NEXT: L..BB1_2:                               # %for.body
; CHECK32-NEXT:                                         # =>This Inner Loop Header: Depth=1
; CHECK32-NEXT:         xscvdpspn vs36, f1
; CHECK32-NEXT:         xxspltiw vs1, 1170469888
; CHECK32-NEXT:         vperm v4, v4, v3, v2
; CHECK32-NEXT:         xvcmpgtsp vs2, vs0, vs36
; CHECK32-NEXT:         xvmaddasp vs1, vs36, vs0
; CHECK32-NEXT:         xxland vs1, vs2, vs1
; CHECK32-NEXT:         xscvspdpn f1, vs1
; CHECK32-NEXT:         b L..BB1_2
