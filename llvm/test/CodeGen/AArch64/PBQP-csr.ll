; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mcpu=cortex-a57 -mattr=+neon -fp-contract=fast -regalloc=pbqp -pbqp-coalescing | FileCheck %s

%pl = type { i32, i32, i32, i32, %p*, %l*, double* }
%p = type { i32, %ca*, [27 x %ca*], %v*, %v*, %v*, i32 }
%ca = type { %v, float, i32 }
%v = type { double, double, double }
%l = type opaque
%rs = type { i32, i32, i32, i32, %v*, %v*, [21 x double], %v, %v, %v, double, double, double }

;CHECK-LABEL: test_csr
define void @test_csr(%pl* nocapture readnone %this, %rs* nocapture %r) align 2 {
;CHECK-NOT: stp {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %x.i = getelementptr inbounds %rs, %rs* %r, i64 0, i32 7, i32 0
  %y.i = getelementptr inbounds %rs, %rs* %r, i64 0, i32 7, i32 1
  %z.i = getelementptr inbounds %rs, %rs* %r, i64 0, i32 7, i32 2
  %x.i61 = getelementptr inbounds %rs, %rs* %r, i64 0, i32 8, i32 0
  %y.i62 = getelementptr inbounds %rs, %rs* %r, i64 0, i32 8, i32 1
  %z.i63 = getelementptr inbounds %rs, %rs* %r, i64 0, i32 8, i32 2
  %x.i58 = getelementptr inbounds %rs, %rs* %r, i64 0, i32 9, i32 0
  %y.i59 = getelementptr inbounds %rs, %rs* %r, i64 0, i32 9, i32 1
  %z.i60 = getelementptr inbounds %rs, %rs* %r, i64 0, i32 9, i32 2
  %na = getelementptr inbounds %rs, %rs* %r, i64 0, i32 0
  %0 = bitcast double* %x.i to i8*
  call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 72, i32 8, i1 false)
  %1 = load i32, i32* %na, align 4
  %cmp70 = icmp sgt i32 %1, 0
  br i1 %cmp70, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %fn = getelementptr inbounds %rs, %rs* %r, i64 0, i32 4
  %2 = load %v*, %v** %fn, align 8
  %fs = getelementptr inbounds %rs, %rs* %r, i64 0, i32 5
  %3 = load %v*, %v** %fs, align 8
  %4 = sext i32 %1 to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %5 = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add6.i, %for.body ]
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %6 = phi <2 x double> [ zeroinitializer, %for.body.lr.ph ], [ %17, %for.body ]
  %7 = phi <2 x double> [ zeroinitializer, %for.body.lr.ph ], [ %22, %for.body ]
  %8 = phi <2 x double> [ zeroinitializer, %for.body.lr.ph ], [ %26, %for.body ]
  %9 = phi <2 x double> [ zeroinitializer, %for.body.lr.ph ], [ %28, %for.body ]
  %x.i54 = getelementptr inbounds %v, %v* %2, i64 %indvars.iv, i32 0
  %x1.i = getelementptr inbounds %v, %v* %3, i64 %indvars.iv, i32 0
  %y.i56 = getelementptr inbounds %v, %v* %2, i64 %indvars.iv, i32 1
  %10 = bitcast double* %x.i54 to <2 x double>*
  %11 = load <2 x double>, <2 x double>* %10, align 8
  %y2.i = getelementptr inbounds %v, %v* %3, i64 %indvars.iv, i32 1
  %12 = bitcast double* %x1.i to <2 x double>*
  %13 = load <2 x double>, <2 x double>* %12, align 8
  %14 = fadd fast <2 x double> %13, %11
  %z.i57 = getelementptr inbounds %v, %v* %2, i64 %indvars.iv, i32 2
  %15 = load double, double* %z.i57, align 8
  %z4.i = getelementptr inbounds %v, %v* %3, i64 %indvars.iv, i32 2
  %16 = load double, double* %z4.i, align 8
  %add5.i = fadd fast double %16, %15
  %17 = fadd fast <2 x double> %6, %11
  %18 = bitcast double* %x.i to <2 x double>*
  store <2 x double> %17, <2 x double>* %18, align 8
  %19 = load double, double* %x1.i, align 8
  %20 = insertelement <2 x double> undef, double %15, i32 0
  %21 = insertelement <2 x double> %20, double %19, i32 1
  %22 = fadd fast <2 x double> %7, %21
  %23 = bitcast double* %z.i to <2 x double>*
  store <2 x double> %22, <2 x double>* %23, align 8
  %24 = bitcast double* %y2.i to <2 x double>*
  %25 = load <2 x double>, <2 x double>* %24, align 8
  %26 = fadd fast <2 x double> %8, %25
  %27 = bitcast double* %y.i62 to <2 x double>*
  store <2 x double> %26, <2 x double>* %27, align 8
  %28 = fadd fast <2 x double> %14, %9
  %29 = bitcast double* %x.i58 to <2 x double>*
  store <2 x double> %28, <2 x double>* %29, align 8
  %add6.i = fadd fast double %add5.i, %5
  store double %add6.i, double* %z.i60, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp slt i64 %indvars.iv.next, %4
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)

