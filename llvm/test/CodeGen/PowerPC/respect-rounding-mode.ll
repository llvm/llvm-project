; The strictfp version of test/CodeGen/PowerPC/cse-despit-rounding-mode.ll
; With strictfp, the MachineIR optimizations need to assume that a call
; can change the rounding mode and must not move/eliminate the repeated
; multiply/convert instructions in this test.
; RUN: llc -verify-machineinstrs --mtriple powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -ppc-asm-full-reg-names < %s | grep 'xvrdpic' | count 4
; RUN: llc -verify-machineinstrs --mtriple powerpc-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -ppc-asm-full-reg-names < %s | grep 'xvrdpic' | count 4
; RUN: llc -verify-machineinstrs --mtriple powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | grep 'xvrdpic' | count 4

; RUN: llc -verify-machineinstrs --mtriple powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -ppc-asm-full-reg-names < %s | grep 'xvmuldp' | count 4
; RUN: llc -verify-machineinstrs --mtriple powerpc-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -ppc-asm-full-reg-names < %s | grep 'xvmuldp' | count 4
; RUN: llc -verify-machineinstrs --mtriple powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | grep 'xvmuldp' | count 4
@IndirectCallPtr = dso_local local_unnamed_addr global void (...)* null, align 8

define dso_local signext i32 @func1() local_unnamed_addr #0 {
entry:
  tail call void bitcast (void (...)* @directCall to void ()*)() #0
  %0 = tail call <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double> <double -9.990000e+01, double 9.990000e+01>, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %vecext = extractelement <2 x double> %0, i32 0
  %sub = tail call double @llvm.experimental.constrained.fsub.f64(double %vecext, double -9.900000e+01, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %conv = tail call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %sub, metadata !"fpexcept.ignore") #0
  tail call void bitcast (void (...)* @directCall to void ()*)() #0
  %1 = tail call <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double> <double -9.990000e+01, double 9.990000e+01>, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %vecext3 = extractelement <2 x double> %1, i32 1
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %vecext3, double 9.900000e+01, metadata !"une", metadata !"fpexcept.ignore") #0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @exit(i32 signext 2) #0
  unreachable

if.end:                                           ; preds = %entry
  ret i32 %conv
}

declare void @directCall(...) local_unnamed_addr

declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)

declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)

declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)

declare void @exit(i32 signext) local_unnamed_addr

define dso_local signext i32 @func2() local_unnamed_addr #0 {
entry:
  %call = tail call <2 x double> bitcast (<2 x double> (...)* @getvector1 to <2 x double> ()*)() #0
  %call1 = tail call <2 x double> bitcast (<2 x double> (...)* @getvector2 to <2 x double> ()*)() #0
  tail call void bitcast (void (...)* @directCall to void ()*)() #0
  %mul = tail call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %call, <2 x double> %call1, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %vecext = extractelement <2 x double> %mul, i32 0
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %vecext, double 4.000000e+00, metadata !"oeq", metadata !"fpexcept.ignore") #0
  br i1 %cmp, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  tail call void bitcast (void (...)* @directCall to void ()*)() #0
  %mul10 = tail call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %call, <2 x double> %call1, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %0 = tail call i32 @llvm.ppc.vsx.xvcmpeqdp.p(i32 2, <2 x double> %mul, <2 x double> %mul10) #0
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %0, %if.end ], [ 11, %entry ]
  ret i32 %retval.0
}

declare <2 x double> @getvector1(...) local_unnamed_addr

declare <2 x double> @getvector2(...) local_unnamed_addr

declare <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double>, <2 x double>, metadata, metadata)

declare i32 @llvm.ppc.vsx.xvcmpeqdp.p(i32, <2 x double>, <2 x double>)

define dso_local signext i32 @func3() local_unnamed_addr #0 {
entry:
  %0 = load void ()*, void ()** bitcast (void (...)** @IndirectCallPtr to void ()**), align 8
  tail call void %0() #0
  %1 = tail call <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double> <double -9.990000e+01, double 9.990000e+01>, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %vecext = extractelement <2 x double> %1, i32 0
  %sub = tail call double @llvm.experimental.constrained.fsub.f64(double %vecext, double -9.900000e+01, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %conv = tail call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %sub, metadata !"fpexcept.ignore") #0
  %2 = load void ()*, void ()** bitcast (void (...)** @IndirectCallPtr to void ()**), align 8
  tail call void %2() #0
  %3 = tail call <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double> <double -9.990000e+01, double 9.990000e+01>, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %vecext4 = extractelement <2 x double> %3, i32 1
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %vecext4, double 9.900000e+01, metadata !"une", metadata !"fpexcept.ignore") #0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @exit(i32 signext 2) #0
  unreachable

if.end:                                           ; preds = %entry
  ret i32 %conv
}

define dso_local signext i32 @func4() local_unnamed_addr #0 {
entry:
  %call = tail call <2 x double> bitcast (<2 x double> (...)* @getvector1 to <2 x double> ()*)() #0
  %call1 = tail call <2 x double> bitcast (<2 x double> (...)* @getvector2 to <2 x double> ()*)() #0
  %0 = load void ()*, void ()** bitcast (void (...)** @IndirectCallPtr to void ()**), align 8
  tail call void %0() #0
  %mul = tail call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %call, <2 x double> %call1, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %vecext = extractelement <2 x double> %mul, i32 0
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %vecext, double 4.000000e+00, metadata !"oeq", metadata !"fpexcept.ignore") #0
  br i1 %cmp, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %1 = load void ()*, void ()** bitcast (void (...)** @IndirectCallPtr to void ()**), align 8
  tail call void %1() #0
  %mul11 = tail call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %call, <2 x double> %call1, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %2 = tail call i32 @llvm.ppc.vsx.xvcmpeqdp.p(i32 2, <2 x double> %mul, <2 x double> %mul11) #0
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %2, %if.end ], [ 11, %entry ]
  ret i32 %retval.0
}

declare <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double>, metadata, metadata)

attributes #0 = { nounwind strictfp }
