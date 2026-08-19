; This incorporates a condition where a qf type generated from widening
; multiplication. The hi/lo of the result is used in an add instruction.
; As a result a COPY instruction is generated to copy the hi/lo bits to
; another virtual register before use in the add instr. For STRICT-IEEE
; mode, we need to convert this to IEEE before use in add instruction,
; and add a new add instr, deleting the original instr. We check for the
; deletion here.
; REQUIRES: asserts
; RUN: llc --mtriple=hexagon-- -O2 -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 -enable-xqf-gen=true -hexagon-qfloat-mode=strict-ieee \
; RUN: -debug-only=hexagon-xqf-gen 2>&1 < %s | FileCheck %s --check-prefix STRICT-IEEE
; RUN: llc --mtriple=hexagon-- -O2 -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 -enable-xqf-gen=true -hexagon-qfloat-mode=ieee \
; RUN: -debug-only=hexagon-xqf-gen 2>&1 < %s | FileCheck %s --check-prefix COMPLIANT-IEEE
; RUN: llc --mtriple=hexagon-- -O2 -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 -enable-xqf-gen=true -hexagon-qfloat-mode=strict-ieee \
; RUN: -debug-only=hexagon-xqf-gen 2>&1 < %s | FileCheck %s --check-prefix STRICT-IEEE
; RUN: llc --mtriple=hexagon-- -O2 -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 -enable-xqf-gen=true -hexagon-qfloat-mode=ieee \
; RUN: -debug-only=hexagon-xqf-gen 2>&1 < %s | FileCheck %s --check-prefix COMPLIANT-IEEE

; STRICT-IEEE: Generating code for STRICT-IEEE mode
; STRICT-IEEE-NEXT: deleting redundant instruction %{{[0-9]+}}:hvxvr = V6_vadd_qf32_mix killed %{{[0-9]+}}:hvxvr, killed %{{[0-9]+}}:hvxvr
; STRICT-IEEE-NEXT: deleting redundant instruction %{{[0-9]+}}:hvxvr = V6_vadd_qf32_mix killed %{{[0-9]+}}:hvxvr, killed %{{[0-9]+}}:hvxvr

; COMPLIANT-IEEE: Generating code for IEEE mode
; COMPLIANT-IEEE-NOT: deleting redundant instruction

define dso_local noundef i32 @main() #0 {
entry:
  tail call void asm sideeffect "l2fetch($0, $1)", "r,r"(ptr blockaddress(@main, %for.cond6.preheader.lr.ph.i), i32 8421392) #4
  %vla69 = alloca [128 x half], align 128
  %vla470 = alloca [32768 x half], align 128
  %vla771 = alloca [256 x half], align 128
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %arrayidx.phi = phi ptr [ %vla69, %entry ], [ %arrayidx.inc, %for.body ]
  %i.077 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %call = tail call i32 @rand() #4
  %rem = srem i32 %call, 10
  %conv = sitofp i32 %rem to double
  %mul17 = fmul double %conv, 5.000000e-01
  %0 = fptrunc double %mul17 to half
  store half %0, ptr %arrayidx.phi, align 2
  %inc = add nuw nsw i32 %i.077, 1
  %exitcond.not = icmp eq i32 %inc, 128
  %arrayidx.inc = getelementptr half, ptr %arrayidx.phi, i32 1
  br i1 %exitcond.not, label %for.body24, label %for.body

for.body24:                                       ; preds = %for.body24, %for.body
  %arrayidx29.phi = phi ptr [ %arrayidx29.inc, %for.body24 ], [ %vla470, %for.body ]
  %i18.078 = phi i32 [ %inc31, %for.body24 ], [ 0, %for.body ]
  %call25 = tail call i32 @rand() #4
  %rem26 = srem i32 %call25, 20
  %conv27 = sitofp i32 %rem26 to double
  %mul28 = fmul double %conv27, 2.500000e-01
  %1 = fptrunc double %mul28 to half
  store half %1, ptr %arrayidx29.phi, align 2
  %inc31 = add nuw nsw i32 %i18.078, 1
  %exitcond79.not = icmp eq i32 %inc31, 32768
  %arrayidx29.inc = getelementptr half, ptr %arrayidx29.phi, i32 1
  br i1 %exitcond79.not, label %for.cond6.preheader.lr.ph.i, label %for.body24

for.cond6.preheader.lr.ph.i:                      ; preds = %for.body24
  tail call void asm sideeffect "labelsym_startofkernel_${:uid}: .global labelsym_startofkernel_${:uid}", ""() #4
  %2 = tail call <64 x i32> @llvm.hexagon.V6.vdd0.128B()
  %3 = tail call <128 x i1> @llvm.hexagon.V6.pred.scalar2v2.128B(i32 128)
  %4 = tail call <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32 0)
  br label %for.body9.i

for.cond.cleanup8.i:                              ; preds = %for.cond.cleanup12.i
  call void asm sideeffect "labelsym_endofkernel_${:uid}: .global labelsym_endofkernel_${:uid}", ""() #4
  ret i32 1

for.body9.i:                                      ; preds = %for.cond.cleanup12.i, %for.cond6.preheader.lr.ph.i
  %arrayidx15.phi.i = phi ptr [ %vla771, %for.cond6.preheader.lr.ph.i ], [ %add.ptr.i.i, %for.cond.cleanup12.i ]
  %storemerge3140.i = phi i32 [ 0, %for.cond6.preheader.lr.ph.i ], [ %add18.i, %for.cond.cleanup12.i ]
  br label %for.body13.i

for.cond.cleanup12.i:                             ; preds = %for.body13.i
  %5 = call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %15)
  %6 = call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %15)
  %7 = call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> %5, <32 x i32> %6)
  %8 = ptrtoint ptr %arrayidx15.phi.i to i32
  %9 = call <32 x i32> @llvm.hexagon.V6.vror.128B(<32 x i32> %7, i32 128)
  %10 = call <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32 %8)
  %11 = call <128 x i1> @llvm.hexagon.V6.pred.and.n.128B(<128 x i1> %3, <128 x i1> %10)
  call void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1> %11, ptr %arrayidx15.phi.i, <32 x i32> %9)
  %add.ptr.i.i = getelementptr i8, ptr %arrayidx15.phi.i, i32 128
  call void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1> %4, ptr nonnull %add.ptr.i.i, <32 x i32> %9)
  %add18.i = add nuw nsw i32 %storemerge3140.i, 64
  %cmp7.i = icmp ult i32 %storemerge3140.i, 192
  br i1 %cmp7.i, label %for.body9.i, label %for.cond.cleanup8.i

for.body13.i:                                     ; preds = %for.body13.i, %for.body9.i
  %12 = phi <64 x i32> [ %15, %for.body13.i ], [ %2, %for.body9.i ]
  %arrayidx4.i.phi.i = phi ptr [ %arrayidx4.i.inc.i, %for.body13.i ], [ %vla69, %for.body9.i ]
  %p.038.i = phi i32 [ %inc.i, %for.body13.i ], [ 0, %for.body9.i ]
  %mul.i.i = shl nsw i32 %p.038.i, 8
  %add.i.i = add nuw nsw i32 %mul.i.i, %storemerge3140.i
  %arrayidx.i.i = getelementptr inbounds i16, ptr %vla470, i32 %add.i.i
  %13 = load <32 x i32>, ptr %arrayidx.i.i, align 128
  %14 = load <128 x i8>, ptr %arrayidx4.i.phi.i, align 1
  %shuffle.i.i = shufflevector <128 x i8> %14, <128 x i8> poison, <128 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  %.cast.i = bitcast <128 x i8> %shuffle.i.i to <32 x i32>
  %15 = call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32> %12, <32 x i32> %.cast.i, <32 x i32> %13)
  %inc.i = add nuw nsw i32 %p.038.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, 128
  %arrayidx4.i.inc.i = getelementptr i16, ptr %arrayidx4.i.phi.i, i32 1
  br i1 %exitcond.not.i, label %for.cond.cleanup12.i, label %for.body13.i
}

; Function Attrs: nounwind
declare dso_local i32 @rand() local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vdd0.128B() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32>, <32 x i32>, <32 x i32>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vror.128B(<32 x i32>, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <128 x i1> @llvm.hexagon.V6.pred.scalar2v2.128B(i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <128 x i1> @llvm.hexagon.V6.pred.and.n.128B(<128 x i1>, <128 x i1>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1>, ptr, <32 x i32>) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32>, <32 x i32>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #2

attributes #0 = { norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8"  "target-features"="+hvx-length128b,-long-calls" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-length128b,-long-calls" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(write) }
attributes #4 = { nounwind }
