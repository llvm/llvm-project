; RUN: llc < %s -mtriple=x86_64-apple-darwin10
; rdar://9045024
; PR9305

define void @calc_gb_rad_still_sse2_double() nounwind ssp {
entry:
  br label %for.cond.outer

for.cond.outer:                                   ; preds = %if.end71, %entry
  %theta.0.ph = phi <2 x double> [ undef, %entry ], [ %theta.1, %if.end71 ]
  %mul.i97 = fmul <2 x double> %theta.0.ph, undef
  %fmul = fmul <2 x double> <double 2.000000e+00, double 2.000000e+00>, undef
  %mul.i96 = fmul <2 x double> %mul.i97, %fmul
  br i1 undef, label %for.body, label %for.end82

for.body:                                         ; preds = %for.cond.outer
  br i1 undef, label %for.body33.lr.ph, label %for.end

for.body33.lr.ph:                                 ; preds = %for.body
  %dccf.2 = select i1 undef, <2 x double> %mul.i96, <2 x double> undef
  unreachable

for.end:                                          ; preds = %for.body
  %vecins.i94 = insertelement <2 x double> undef, double 0.000000e+00, i32 0
  %cmpsd.i = tail call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %vecins.i94, <2 x double> <double 0x3FE984B204153B34, double 0x3FE984B204153B34>, i8 2) nounwind
  tail call void (...) @_mm_movemask_pd(<2 x double> %cmpsd.i) nounwind
  br i1 undef, label %if.then67, label %if.end71

if.then67:                                        ; preds = %for.end
  %vecins.i91 = insertelement <2 x double> %vecins.i94, double undef, i32 0
  br label %if.end71

if.end71:                                         ; preds = %if.then67, %for.end
  %theta.1 = phi <2 x double> [ %vecins.i91, %if.then67 ], [ %theta.0.ph, %for.end ]
  br label %for.cond.outer

for.end82:                                        ; preds = %for.cond.outer
  ret void
}

declare void @_mm_movemask_pd(...)

declare <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double>, <2 x double>, i8) nounwind readnone
