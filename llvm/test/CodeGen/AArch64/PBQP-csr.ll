; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mcpu=cortex-a57 -mattr=+neon -fp-contract=fast -regalloc=pbqp -pbqp-coalescing | FileCheck %s

%pl = type { i32, i32, i32, i32, ptr, ptr, ptr }
%p = type { i32, ptr, [27 x ptr], ptr, ptr, ptr, i32 }
%ca = type { %v, float, i32 }
%v = type { double, double, double }
%l = type opaque
%rs = type { i32, i32, i32, i32, ptr, ptr, [21 x double], %v, %v, %v, double, double, double }

;CHECK-LABEL: test_csr
define void @test_csr(ptr nocapture readnone %this, ptr nocapture %r) align 2 {
;CHECK-NOT: stp {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %x.i = getelementptr inbounds %rs, ptr %r, i64 0, i32 7, i32 0
  %y.i = getelementptr inbounds %rs, ptr %r, i64 0, i32 7, i32 1
  %z.i = getelementptr inbounds %rs, ptr %r, i64 0, i32 7, i32 2
  %x.i61 = getelementptr inbounds %rs, ptr %r, i64 0, i32 8, i32 0
  %y.i62 = getelementptr inbounds %rs, ptr %r, i64 0, i32 8, i32 1
  %z.i63 = getelementptr inbounds %rs, ptr %r, i64 0, i32 8, i32 2
  %x.i58 = getelementptr inbounds %rs, ptr %r, i64 0, i32 9, i32 0
  %y.i59 = getelementptr inbounds %rs, ptr %r, i64 0, i32 9, i32 1
  %z.i60 = getelementptr inbounds %rs, ptr %r, i64 0, i32 9, i32 2
  %na = getelementptr inbounds %rs, ptr %r, i64 0, i32 0
  %0 = bitcast ptr %x.i to ptr
  call void @llvm.memset.p0.i64(ptr align 8 %0, i8 0, i64 72, i1 false)
  %1 = load i32, ptr %na, align 4
  %cmp70 = icmp sgt i32 %1, 0
  br i1 %cmp70, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %fn = getelementptr inbounds %rs, ptr %r, i64 0, i32 4
  %2 = load ptr, ptr %fn, align 8
  %fs = getelementptr inbounds %rs, ptr %r, i64 0, i32 5
  %3 = load ptr, ptr %fs, align 8
  %4 = sext i32 %1 to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %5 = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add6.i, %for.body ]
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %6 = phi <2 x double> [ zeroinitializer, %for.body.lr.ph ], [ %17, %for.body ]
  %7 = phi <2 x double> [ zeroinitializer, %for.body.lr.ph ], [ %22, %for.body ]
  %8 = phi <2 x double> [ zeroinitializer, %for.body.lr.ph ], [ %26, %for.body ]
  %9 = phi <2 x double> [ zeroinitializer, %for.body.lr.ph ], [ %28, %for.body ]
  %x.i54 = getelementptr inbounds %v, ptr %2, i64 %indvars.iv, i32 0
  %x1.i = getelementptr inbounds %v, ptr %3, i64 %indvars.iv, i32 0
  %y.i56 = getelementptr inbounds %v, ptr %2, i64 %indvars.iv, i32 1
  %10 = bitcast ptr %x.i54 to ptr
  %11 = load <2 x double>, ptr %10, align 8
  %y2.i = getelementptr inbounds %v, ptr %3, i64 %indvars.iv, i32 1
  %12 = bitcast ptr %x1.i to ptr
  %13 = load <2 x double>, ptr %12, align 8
  %14 = fadd fast <2 x double> %13, %11
  %z.i57 = getelementptr inbounds %v, ptr %2, i64 %indvars.iv, i32 2
  %15 = load double, ptr %z.i57, align 8
  %z4.i = getelementptr inbounds %v, ptr %3, i64 %indvars.iv, i32 2
  %16 = load double, ptr %z4.i, align 8
  %add5.i = fadd fast double %16, %15
  %17 = fadd fast <2 x double> %6, %11
  %18 = bitcast ptr %x.i to ptr
  store <2 x double> %17, ptr %18, align 8
  %19 = load double, ptr %x1.i, align 8
  %20 = insertelement <2 x double> undef, double %15, i32 0
  %21 = insertelement <2 x double> %20, double %19, i32 1
  %22 = fadd fast <2 x double> %7, %21
  %23 = bitcast ptr %z.i to ptr
  store <2 x double> %22, ptr %23, align 8
  %24 = bitcast ptr %y2.i to ptr
  %25 = load <2 x double>, ptr %24, align 8
  %26 = fadd fast <2 x double> %8, %25
  %27 = bitcast ptr %y.i62 to ptr
  store <2 x double> %26, ptr %27, align 8
  %28 = fadd fast <2 x double> %14, %9
  %29 = bitcast ptr %x.i58 to ptr
  store <2 x double> %28, ptr %29, align 8
  %add6.i = fadd fast double %add5.i, %5
  store double %add6.i, ptr %z.i60, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp slt i64 %indvars.iv.next, %4
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)

