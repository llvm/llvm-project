; Check that indvars transforms Tapir loops to use backedges based on
; equality comparions, even if the transformation might be high cost.
;
; RUN: opt < %s -indvars -S | FileCheck %s
; RUN: opt < %s -passes='indvars' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._seq = type { i64*, i64 }

$_ZN8sequence4packIllN5utils9identityFIlEELi2048EEE4_seqIT_EPS5_PbT0_S9_T1_ = comdat any

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #5

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #7

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #8

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #2

; Function Attrs: uwtable
define linkonce_odr dso_local { i64*, i64 } @_ZN8sequence4packIllN5utils9identityFIlEELi2048EEE4_seqIT_EPS5_PbT0_S9_T1_(i64* %Out, i8* %Fl, i64 %s, i64 %e) local_unnamed_addr #0 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %retval = alloca %struct._seq, align 8
  %_ee = alloca i64, align 8
  %_ee49 = alloca i64, align 8
  %syncreg56 = tail call token @llvm.syncregion.start()
  %0 = xor i64 %s, -1
  %sub1 = add i64 %0, %e
  %div = sdiv i64 %sub1, 2048
  %add = add nsw i64 %div, 1
  %cmp = icmp slt i64 %add, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = tail call { i64*, i64 } @_ZN8sequence10packSerialIllN5utils9identityFIlEEEE4_seqIT_EPS5_PbT0_S9_T1_(i64* %Out, i8* %Fl, i64 %s, i64 %e)
  %1 = getelementptr inbounds %struct._seq, %struct._seq* %retval, i64 0, i32 0
  %2 = extractvalue { i64*, i64 } %call, 0
  store i64* %2, i64** %1, align 8
  %3 = getelementptr inbounds %struct._seq, %struct._seq* %retval, i64 0, i32 1
  %4 = extractvalue { i64*, i64 } %call, 1
  store i64 %4, i64* %3, align 8
  br label %cleanup129

if.end:                                           ; preds = %entry
  %mul = shl nsw i64 %add, 3
  %call2 = tail call noalias i8* @malloc(i64 %mul) #23
  %5 = bitcast i8* %call2 to i64*
  %6 = bitcast i64* %_ee to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6) #23
  store i64 %e, i64* %_ee, align 8, !tbaa !17
  br label %pfor.cond

pfor.cond:                                        ; preds = %pfor.inc, %if.end
  %__begin.0 = phi i64 [ 0, %if.end ], [ %inc, %pfor.inc ]
  detach within %syncreg56, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.cond
  %ref.tmp = alloca i64, align 8
  %mul15 = shl nsw i64 %__begin.0, 11
  %add16 = add nsw i64 %mul15, %s
  %7 = bitcast i64* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %7) #23
  %add18 = add nsw i64 %add16, 2048
  store i64 %add18, i64* %ref.tmp, align 8, !tbaa !17
  %call19 = call dereferenceable(8) i64* @_ZSt3minIlERKT_S2_S2_(i64* nonnull dereferenceable(8) %ref.tmp, i64* nonnull dereferenceable(8) %_ee)
  %8 = load i64, i64* %call19, align 8, !tbaa !17
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %7) #23
  %add.ptr = getelementptr inbounds i8, i8* %Fl, i64 %add16
  %sub20 = sub nsw i64 %8, %add16
  %call23 = call i64 @_ZN8sequence14sumFlagsSerialIlEET_PbS1_(i8* %add.ptr, i64 %sub20)
  %arrayidx = getelementptr inbounds i64, i64* %5, i64 %__begin.0
  store i64 %call23, i64* %arrayidx, align 8, !tbaa !17
  reattach within %syncreg56, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.cond, %pfor.body
  %inc = add nuw nsw i64 %__begin.0, 1
  %exitcond = icmp eq i64 %inc, %add
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.cond, !llvm.loop !78

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg56, label %cleanup

cleanup:                                          ; preds = %pfor.cond.cleanup
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6) #23
  %call42 = call i64 @_ZN8sequence8plusScanIllLi2048EEET_PS1_S2_T0_(i64* %5, i64* %5, i64 %add)
  %cmp43 = icmp eq i64* %Out, null
  br i1 %cmp43, label %if.then44, label %if.end47

if.then44:                                        ; preds = %cleanup
  %mul45 = shl i64 %call42, 3
  %call46 = call noalias i8* @malloc(i64 %mul45) #23
  %9 = bitcast i8* %call46 to i64*
  br label %if.end47

if.end47:                                         ; preds = %if.then44, %cleanup
  %Out.addr.0 = phi i64* [ %9, %if.then44 ], [ %Out, %cleanup ]
  %10 = bitcast i64* %_ee49 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %10) #23
  store i64 %e, i64* %_ee49, align 8, !tbaa !17
  br label %pfor.cond68

; CHECK: if.end47:
; CHECK: %[[ICMPDIV:.+]] = icmp sgt i64 %div, 0
; CHECK-NEXT: %[[SMAX:.+]] = select i1 %[[ICMPDIV]], i64 %div, i64 0
; CHECK-NEXT: %[[LIMIT:.+]] = add nuw nsw i64 %[[SMAX]], 1
; CHECK-NEXT: br label %pfor.cond68

pfor.cond68:                                      ; preds = %pfor.inc97, %if.end47
  %__begin62.0 = phi i64 [ 0, %if.end47 ], [ %inc98, %pfor.inc97 ]
  detach within %syncreg56, label %pfor.body74, label %pfor.inc97 unwind label %lpad99.loopexit

pfor.body74:                                      ; preds = %pfor.cond68
  %ref.tmp79 = alloca i64, align 8
  %mul76 = shl nsw i64 %__begin62.0, 11
  %add77 = add nsw i64 %mul76, %s
  %11 = bitcast i64* %ref.tmp79 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %11) #23
  %add80 = add nsw i64 %add77, 2048
  store i64 %add80, i64* %ref.tmp79, align 8, !tbaa !17
  %call85 = call dereferenceable(8) i64* @_ZSt3minIlERKT_S2_S2_(i64* nonnull dereferenceable(8) %ref.tmp79, i64* nonnull dereferenceable(8) %_ee49)
  %12 = load i64, i64* %call85, align 8, !tbaa !17
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %11) #23
  %arrayidx87 = getelementptr inbounds i64, i64* %5, i64 %__begin62.0
  %13 = load i64, i64* %arrayidx87, align 8, !tbaa !17
  %add.ptr88 = getelementptr inbounds i64, i64* %Out.addr.0, i64 %13
  %call92 = invoke { i64*, i64 } @_ZN8sequence10packSerialIllN5utils9identityFIlEEEE4_seqIT_EPS5_PbT0_S9_T1_(i64* %add.ptr88, i8* %Fl, i64 %add77, i64 %12)
          to label %invoke.cont91 unwind label %lpad90

invoke.cont91:                                    ; preds = %pfor.body74
  reattach within %syncreg56, label %pfor.inc97

pfor.inc97:                                       ; preds = %pfor.cond68, %invoke.cont91
  %inc98 = add nuw nsw i64 %__begin62.0, 1
  %cmp106 = icmp slt i64 %__begin62.0, %div
  br i1 %cmp106, label %pfor.cond68, label %pfor.cond.cleanup107, !llvm.loop !79

; CHECK: pfor.inc97:
; CHECK-NEXT: %inc98 = add nuw nsw i64 %__begin62.0, 1
; CHECK-NEXT: %[[CMP:.+]] = icmp ne i64 %inc98, %[[LIMIT]]
; CHECK-NEXT: br i1 %[[CMP]], label %pfor.cond68, label %pfor.cond.cleanup107

pfor.cond.cleanup107:                             ; preds = %pfor.inc97
  sync within %syncreg56, label %cleanup116

lpad90:                                           ; preds = %pfor.body74
  %14 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg56, { i8*, i32 } %14)
          to label %det.rethrow.unreachable105 unwind label %lpad99.loopexit.split-lp

det.rethrow.unreachable105:                       ; preds = %lpad90
  unreachable

lpad99.loopexit:                                  ; preds = %pfor.cond68
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad99

lpad99.loopexit.split-lp:                         ; preds = %lpad90
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad99

lpad99:                                           ; preds = %lpad99.loopexit.split-lp, %lpad99.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad99.loopexit ], [ %lpad.loopexit.split-lp, %lpad99.loopexit.split-lp ]
  sync within %syncreg56, label %sync.continue111

sync.continue111:                                 ; preds = %lpad99
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %10) #23
  resume { i8*, i32 } %lpad.phi

cleanup116:                                       ; preds = %pfor.cond.cleanup107
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %10) #23
  call void @free(i8* %call2) #23
  call void @_ZN4_seqIlEC2EPll(%struct._seq* nonnull %retval, i64* %Out.addr.0, i64 %call42)
  br label %cleanup129

cleanup129:                                       ; preds = %cleanup116, %if.then
  %.fca.0.gep = getelementptr inbounds %struct._seq, %struct._seq* %retval, i64 0, i32 0
  %.fca.0.load = load i64*, i64** %.fca.0.gep, align 8
  %.fca.0.insert = insertvalue { i64*, i64 } undef, i64* %.fca.0.load, 0
  %15 = getelementptr inbounds %struct._seq, %struct._seq* %retval, i64 0, i32 1
  %.fca.1.load = load i64, i64* %15, align 8
  %.fca.1.insert = insertvalue { i64*, i64 } %.fca.0.insert, i64 %.fca.1.load, 1
  ret { i64*, i64 } %.fca.1.insert
}

; Function Attrs: uwtable
declare dso_local { i64*, i64 } @_ZN8sequence10packSerialIllN5utils9identityFIlEEEE4_seqIT_EPS5_PbT0_S9_T1_(i64* %Out, i8* %Fl, i64 %s, i64 %e) local_unnamed_addr #0

; Function Attrs: inlinehint nounwind uwtable
declare dso_local dereferenceable(8) i64* @_ZSt3minIlERKT_S2_S2_(i64* dereferenceable(8) %__a, i64* dereferenceable(8) %__b) local_unnamed_addr #6

; Function Attrs: nounwind uwtable
declare dso_local i64 @_ZN8sequence14sumFlagsSerialIlEET_PbS1_(i8* %Fl, i64 %n) local_unnamed_addr #4

; Function Attrs: uwtable
declare dso_local i64 @_ZN8sequence8plusScanIllLi2048EEET_PS1_S2_T0_(i64* %In, i64* %Out, i64 %n) local_unnamed_addr #0

; Function Attrs: nounwind uwtable
declare dso_local void @_ZN4_seqIlEC2EPll(%struct._seq* %this, i64* %_A, i64 %_n) unnamed_addr #4

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="skylake-avx512" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+mpx,+pclmul,+pku,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-avx512bf16,-avx512bitalg,-avx512er,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vnni,-avx512vpopcntdq,-cldemote,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-prefetchwt1,-ptwrite,-rdpid,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="skylake-avx512" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+mpx,+pclmul,+pku,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-avx512bf16,-avx512bitalg,-avx512er,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vnni,-avx512vpopcntdq,-cldemote,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-prefetchwt1,-ptwrite,-rdpid,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="skylake-avx512" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+mpx,+pclmul,+pku,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-avx512bf16,-avx512bitalg,-avx512er,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vnni,-avx512vpopcntdq,-cldemote,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-prefetchwt1,-ptwrite,-rdpid,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="skylake-avx512" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+mpx,+pclmul,+pku,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-avx512bf16,-avx512bitalg,-avx512er,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vnni,-avx512vpopcntdq,-cldemote,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-prefetchwt1,-ptwrite,-rdpid,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { argmemonly }
attributes #8 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="skylake-avx512" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+mpx,+pclmul,+pku,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-avx512bf16,-avx512bitalg,-avx512er,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vnni,-avx512vpopcntdq,-cldemote,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-prefetchwt1,-ptwrite,-rdpid,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #23 = { nounwind }

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!14 = !{!"tapir.loop.spawn.strategy", i32 1}
!17 = !{!18, !18, i64 0}
!18 = !{!"long", !5, i64 0}
!78 = distinct !{!78, !14}
!79 = distinct !{!79, !14}
