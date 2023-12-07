; RUN: llc -mtriple arm64-apple-ios -O3 -o - < %s | FileCheck %s
; <rdar://problem/14477220>

%class.Complex = type { float, float }
%class.Complex_int = type { i32, i32 }
%class.Complex_long = type { i64, i64 }

; CHECK-LABEL: @test
; CHECK: add [[BASE:x[0-9]+]], x0, x1, lsl #3
; CHECK: ldp [[CPLX1_I:s[0-9]+]], [[CPLX1_R:s[0-9]+]], [[[BASE]]]
; CHECK: ldp [[CPLX2_I:s[0-9]+]], [[CPLX2_R:s[0-9]+]], [[[BASE]], #64]
; CHECK: fadd {{s[0-9]+}}, [[CPLX2_I]], [[CPLX1_I]]
; CHECK: fadd {{s[0-9]+}}, [[CPLX2_R]], [[CPLX1_R]]
; CHECK: ret
define void @test(ptr nocapture %out, i64 %out_start) {
entry:
  %arrayidx = getelementptr inbounds %class.Complex, ptr %out, i64 %out_start
  %0 = load i64, ptr %arrayidx, align 4
  %t0.sroa.0.0.extract.trunc = trunc i64 %0 to i32
  %1 = bitcast i32 %t0.sroa.0.0.extract.trunc to float
  %t0.sroa.2.0.extract.shift = lshr i64 %0, 32
  %t0.sroa.2.0.extract.trunc = trunc i64 %t0.sroa.2.0.extract.shift to i32
  %2 = bitcast i32 %t0.sroa.2.0.extract.trunc to float
  %add = add i64 %out_start, 8
  %arrayidx2 = getelementptr inbounds %class.Complex, ptr %out, i64 %add
  %3 = load float, ptr %arrayidx2, align 4
  %add.i = fadd float %3, %1
  %retval.sroa.0.0.vec.insert.i = insertelement <2 x float> undef, float %add.i, i32 0
  %r.i = getelementptr inbounds %class.Complex, ptr %arrayidx2, i64 0, i32 1
  %4 = load float, ptr %r.i, align 4
  %add5.i = fadd float %4, %2
  %retval.sroa.0.4.vec.insert.i = insertelement <2 x float> %retval.sroa.0.0.vec.insert.i, float %add5.i, i32 1
  store <2 x float> %retval.sroa.0.4.vec.insert.i, ptr %arrayidx, align 4
  ret void
}

; CHECK-LABEL: @test_int
; CHECK: add [[BASE:x[0-9]+]], x0, x1, lsl #3
; CHECK: ldp [[CPLX1_I:w[0-9]+]], [[CPLX1_R:w[0-9]+]], [[[BASE]]]
; CHECK: ldp [[CPLX2_I:w[0-9]+]], [[CPLX2_R:w[0-9]+]], [[[BASE]], #64]
; CHECK: add {{w[0-9]+}}, [[CPLX2_I]], [[CPLX1_I]]
; CHECK: add {{w[0-9]+}}, [[CPLX2_R]], [[CPLX1_R]]
; CHECK: ret
define void @test_int(ptr nocapture %out, i64 %out_start) {
entry:
  %arrayidx = getelementptr inbounds %class.Complex_int, ptr %out, i64 %out_start
  %0 = load i64, ptr %arrayidx, align 4
  %t0.sroa.0.0.extract.trunc = trunc i64 %0 to i32
  %1 = bitcast i32 %t0.sroa.0.0.extract.trunc to i32
  %t0.sroa.2.0.extract.shift = lshr i64 %0, 32
  %t0.sroa.2.0.extract.trunc = trunc i64 %t0.sroa.2.0.extract.shift to i32
  %2 = bitcast i32 %t0.sroa.2.0.extract.trunc to i32
  %add = add i64 %out_start, 8
  %arrayidx2 = getelementptr inbounds %class.Complex_int, ptr %out, i64 %add
  %3 = load i32, ptr %arrayidx2, align 4
  %add.i = add i32 %3, %1
  %retval.sroa.0.0.vec.insert.i = insertelement <2 x i32> undef, i32 %add.i, i32 0
  %r.i = getelementptr inbounds %class.Complex_int, ptr %arrayidx2, i64 0, i32 1
  %4 = load i32, ptr %r.i, align 4
  %add5.i = add i32 %4, %2
  %retval.sroa.0.4.vec.insert.i = insertelement <2 x i32> %retval.sroa.0.0.vec.insert.i, i32 %add5.i, i32 1
  store <2 x i32> %retval.sroa.0.4.vec.insert.i, ptr %arrayidx, align 4
  ret void
}

; CHECK-LABEL: @test_long
; CHECK: add [[BASE:x[0-9]+]], x0, x1, lsl #4
; CHECK: ldp [[CPLX1_I:x[0-9]+]], [[CPLX1_R:x[0-9]+]], [[[BASE]]]
; CHECK: ldp [[CPLX2_I:x[0-9]+]], [[CPLX2_R:x[0-9]+]], [[[BASE]], #128]
; CHECK: add {{x[0-9]+}}, [[CPLX2_I]], [[CPLX1_I]]
; CHECK: add {{x[0-9]+}}, [[CPLX2_R]], [[CPLX1_R]]
; CHECK: ret
define void @test_long(ptr nocapture %out, i64 %out_start) {
entry:
  %arrayidx = getelementptr inbounds %class.Complex_long, ptr %out, i64 %out_start
  %0 = load i128, ptr %arrayidx, align 4
  %t0.sroa.0.0.extract.trunc = trunc i128 %0 to i64
  %1 = bitcast i64 %t0.sroa.0.0.extract.trunc to i64
  %t0.sroa.2.0.extract.shift = lshr i128 %0, 64
  %t0.sroa.2.0.extract.trunc = trunc i128 %t0.sroa.2.0.extract.shift to i64
  %2 = bitcast i64 %t0.sroa.2.0.extract.trunc to i64
  %add = add i64 %out_start, 8
  %arrayidx2 = getelementptr inbounds %class.Complex_long, ptr %out, i64 %add
  %3 = load i64, ptr %arrayidx2, align 4
  %add.i = add i64 %3, %1
  %retval.sroa.0.0.vec.insert.i = insertelement <2 x i64> undef, i64 %add.i, i32 0
  %r.i = getelementptr inbounds %class.Complex_long, ptr %arrayidx2, i32 0, i32 1
  %4 = load i64, ptr %r.i, align 4
  %add5.i = add i64 %4, %2
  %retval.sroa.0.4.vec.insert.i = insertelement <2 x i64> %retval.sroa.0.0.vec.insert.i, i64 %add5.i, i32 1
  store <2 x i64> %retval.sroa.0.4.vec.insert.i, ptr %arrayidx, align 4
  ret void
}
