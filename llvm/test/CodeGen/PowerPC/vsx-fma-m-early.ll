; RUN: llc -verify-machineinstrs -mcpu=pwr10 -disable-ppc-vsx-fma-mutation=false \
; RUN:   -ppc-asm-full-reg-names -schedule-ppc-vsx-fma-mutation-early \
; RUN:    -mtriple powerpc64-ibm-aix7.2.0.0 < %s | FileCheck --check-prefix=CHECK-M %s

; RUN: llc -verify-machineinstrs -mcpu=pwr10 -disable-ppc-vsx-fma-mutation=false \
; RUN:   -ppc-asm-full-reg-names -mtriple powerpc64-ibm-aix7.2.0.0 < %s | \
; RUN:   FileCheck --check-prefix=CHECK-A %s

define void @vsexp(ptr noalias nocapture noundef writeonly %__output_a, ptr noalias nocapture noundef readonly %var1321In_a, ptr noalias nocapture noundef readonly %n) {
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

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) 

; CHECK-M:              .csect ..text..[PR],5{{[[:space:]].*}}.vsexp: 
; CHECK-M-NEXT: # %bb.0:                                # %entry
; CHECK-M-NEXT:         lwz r5, 0(r5)
; CHECK-M-NEXT:         cmpwi   r5, 1
; CHECK-M-NEXT:         bltlr   cr0
; CHECK-M-NEXT: # %bb.1:                                # %for.body.preheader
; CHECK-M-NEXT:         xxspltiw vs0, 1069066811
; CHECK-M-NEXT:         xxspltiw vs1, 1170469888
; CHECK-M-NEXT:         mtctr r5
; CHECK-M-NEXT:         li r5, 0
; CHECK-M-NEXT:         .align  5
; CHECK-M-NEXT: L..BB0_2:                               # %for.body
; CHECK-M-NEXT:                                         # =>This Inner Loop Header: Depth=1
; CHECK-M-NEXT:         lxvx vs2, r4, r5
; CHECK-M-NEXT:         xvmaddmsp vs2, vs0, vs1
; CHECK-M-NEXT:         stxvx vs2, r3, r5
; CHECK-M-NEXT:         addi r5, r5, 16
; CHECK-M-NEXT:         bdnz L..BB0_2
; CHECK-M-NEXT: # %bb.3:                                # %for.end
; CHECK-M-NEXT:         blr

; CHECK-A:              .csect ..text..[PR],5{{[[:space:]].*}}.vsexp:
; CHECK-A-NEXT: # %bb.0:                                # %entry
; CHECK-A-NEXT:         lwz r5, 0(r5)
; CHECK-A-NEXT:         cmpwi   r5, 1
; CHECK-A-NEXT:         bltlr   cr0
; CHECK-A-NEXT: # %bb.1:                                # %for.body.preheader
; CHECK-A-NEXT:         xxspltiw vs0, 1069066811
; CHECK-A-NEXT:         mtctr r5
; CHECK-A-NEXT:         li r5, 0
; CHECK-A-NEXT:         .align  5
; CHECK-A-NEXT: L..BB0_2:                               # %for.body
; CHECK-A-NEXT:                                         # =>This Inner Loop Header: Depth=1
; CHECK-A-NEXT:         lxvx vs1, r4, r5
; CHECK-A-NEXT:         xxspltiw vs2, 1170469888
; CHECK-A-NEXT:         xvmaddasp vs2, vs1, vs0
; CHECK-A-NEXT:         stxvx vs2, r3, r5
; CHECK-A-NEXT:         addi r5, r5, 16
; CHECK-A-NEXT:         bdnz L..BB0_2
; CHECK-A-NEXT: # %bb.3:                                # %for.end
; CHECK-A-NEXT:         blr
