; RUN: opt < %s -constprop -S | FileCheck %s
; REQUIRES: x86-registered-target

define i1 @test_avx512_cvts_exact() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_exact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2si32(<4 x float> <float 3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2si64(<4 x float> <float 3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> <double 7.0, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2si64(<2 x double> <double 7.0, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 10
  %cmp13 = icmp eq i64 %sum13, 10
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

define i1 @test_avx512_cvts_exact_max() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_exact_max(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> <double 2147483647.0, double undef>, i32 4) nounwind
  %b = icmp eq i32 %i0, 2147483647
  ret i1 %b
}

define i1 @test_avx512_cvts_exact_max_p1() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_exact_max_p1(
; CHECK: call
entry:
  %i0 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> <double 2147483648.0, double undef>, i32 4) nounwind
  %b = icmp eq i32 %i0, 2147483648
  ret i1 %b
}

define i1 @test_avx512_cvts_exact_neg_max() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_exact_neg_max(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> <double -2147483648.0, double undef>, i32 4) nounwind
  %b = icmp eq i32 %i0, -2147483648
  ret i1 %b
}

define i1 @test_avx512_cvts_exact_neg_max_p1() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_exact_neg_max_p1(
; CHECK: call
entry:
  %i0 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> <double -2147483649.0, double undef>, i32 4) nounwind
  %b = icmp eq i32 %i0, -2147483649
  ret i1 %b
}

; Inexact values should not fold as they are dependent on rounding mode
define i1 @test_avx512_cvts_inexact() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_inexact(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2si32(<4 x float> <float 1.75, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2si64(<4 x float> <float 1.75, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> <double 1.75, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2si64(<2 x double> <double 1.75, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 4
  %cmp13 = icmp eq i64 %sum13, 4
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

; FLT_MAX/DBL_MAX should not fold
define i1 @test_avx512_cvts_max() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_max(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095039, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405311, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2si32(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2si64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2si64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; INF should not fold
define i1 @test_avx512_cvts_inf() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_inf(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095040, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405312, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2si32(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2si64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2si64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; NAN should not fold
define i1 @test_avx512_cvts_nan() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvts_nan(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2143289344, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9221120237041090560, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2si32(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2si64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2si64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

define i1 @test_avx512_cvtts_exact() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtts_exact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2si(<4 x float> <float 3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2si64(<4 x float> <float 3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2si(<2 x double> <double 7.0, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2si64(<2 x double> <double 7.0, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 10
  %cmp13 = icmp eq i64 %sum13, 10
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

define i1 @test_avx512_cvtts_inexact() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtts_inexact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2si(<4 x float> <float 1.75, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2si64(<4 x float> <float 1.75, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2si(<2 x double> <double 1.75, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2si64(<2 x double> <double 1.75, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 2
  %cmp13 = icmp eq i64 %sum13, 2
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

; FLT_MAX/DBL_MAX should not fold
define i1 @test_avx512_cvtts_max() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtts_max(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095039, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405311, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2si(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2si64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2si(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2si64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; INF should not fold
define i1 @test_avx512_cvtts_inf() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtts_inf(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095040, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405312, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2si(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2si64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2si(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2si64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; NAN should not fold
define i1 @test_avx512_cvtts_nan() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtts_nan(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2143289344, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9221120237041090560, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2si(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2si64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2si(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2si64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

define i1 @test_avx512_cvtu_exact() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtu_exact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2usi32(<4 x float> <float 3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2usi64(<4 x float> <float 3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double> <double 7.0, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2usi64(<2 x double> <double 7.0, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 10
  %cmp13 = icmp eq i64 %sum13, 10
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

; Negative values should not fold as they can't be represented in an unsigned int.
define i1 @test_avx512_cvtu_neg() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtu_neg(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2usi32(<4 x float> <float -3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2usi64(<4 x float> <float -3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double> <double -7.0, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2usi64(<2 x double> <double -7.0, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, -10
  %cmp13 = icmp eq i64 %sum13, -10
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

define i1 @test_avx512_cvtu_exact_max() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtu_exact_max(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = call i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double> <double 4294967295.0, double undef>, i32 4) nounwind
  %b = icmp eq i32 %i0, 4294967295
  ret i1 %b
}

define i1 @test_avx512_cvtu_exact_max_p1() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtu_exact_max_p1(
; CHECK: call
entry:
  %i0 = call i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double> <double 4294967296.0, double undef>, i32 4) nounwind
  %b = icmp eq i32 %i0, 4294967296
  ret i1 %b
}

; Inexact values should not fold as they are dependent on rounding mode
define i1 @test_avx512_cvtu_inexact() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtu_inexact(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2usi32(<4 x float> <float 1.75, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2usi64(<4 x float> <float 1.75, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double> <double 1.75, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2usi64(<2 x double> <double 1.75, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 4
  %cmp13 = icmp eq i64 %sum13, 4
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

; FLT_MAX/DBL_MAX should not fold
define i1 @test_avx512_cvtu_max() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtu_max(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095039, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405311, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2usi32(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2usi64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2usi64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; INF should not fold
define i1 @test_avx512_cvtu_inf() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtu_inf(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095040, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405312, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2usi32(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2usi64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2usi64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; NAN should not fold
define i1 @test_avx512_cvtu_nan() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvtu_nan(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2143289344, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9221120237041090560, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.vcvtss2usi32(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.vcvtss2usi64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.vcvtsd2usi64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

define i1 @test_avx512_cvttu_exact() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvttu_exact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2usi(<4 x float> <float 3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2usi64(<4 x float> <float 3.0, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2usi(<2 x double> <double 7.0, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double> <double 7.0, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 10
  %cmp13 = icmp eq i64 %sum13, 10
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

define i1 @test_avx512_cvttu_inexact() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvttu_inexact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2usi(<4 x float> <float 1.75, float undef, float undef, float undef>, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2usi64(<4 x float> <float 1.75, float undef, float undef, float undef>, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2usi(<2 x double> <double 1.75, double undef>, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double> <double 1.75, double undef>, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 2
  %cmp13 = icmp eq i64 %sum13, 2
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

; FLT_MAX/DBL_MAX should not fold
define i1 @test_avx512_cvttu_max() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvttu_max(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095039, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405311, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2usi(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2usi64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2usi(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; INF should not fold
define i1 @test_avx512_cvttu_inf() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvttu_inf(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095040, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405312, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2usi(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2usi64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2usi(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; NAN should not fold
define i1 @test_avx512_cvttu_nan() nounwind readnone {
; CHECK-LABEL: @test_avx512_cvttu_nan(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2143289344, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9221120237041090560, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.avx512.cvttss2usi(<4 x float> %fm, i32 4) nounwind
  %i1 = tail call i64 @llvm.x86.avx512.cvttss2usi64(<4 x float> %fm, i32 4) nounwind
  %i2 = call i32 @llvm.x86.avx512.cvttsd2usi(<2 x double> %dm, i32 4) nounwind
  %i3 = call i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double> %dm, i32 4) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

declare i32 @llvm.x86.avx512.vcvtss2si32(<4 x float>, i32) nounwind readnone
declare i32 @llvm.x86.avx512.cvttss2si(<4 x float>, i32) nounwind readnone
declare i64 @llvm.x86.avx512.vcvtss2si64(<4 x float>, i32) nounwind readnone
declare i64 @llvm.x86.avx512.cvttss2si64(<4 x float>, i32) nounwind readnone
declare i32 @llvm.x86.avx512.vcvtsd2si32(<2 x double>, i32) nounwind readnone
declare i32 @llvm.x86.avx512.cvttsd2si(<2 x double>, i32) nounwind readnone
declare i64 @llvm.x86.avx512.vcvtsd2si64(<2 x double>, i32) nounwind readnone
declare i64 @llvm.x86.avx512.cvttsd2si64(<2 x double>, i32) nounwind readnone
declare i32 @llvm.x86.avx512.vcvtss2usi32(<4 x float>, i32) nounwind readnone
declare i32 @llvm.x86.avx512.cvttss2usi(<4 x float>, i32) nounwind readnone
declare i64 @llvm.x86.avx512.vcvtss2usi64(<4 x float>, i32) nounwind readnone
declare i64 @llvm.x86.avx512.cvttss2usi64(<4 x float>, i32) nounwind readnone
declare i32 @llvm.x86.avx512.vcvtsd2usi32(<2 x double>, i32) nounwind readnone
declare i32 @llvm.x86.avx512.cvttsd2usi(<2 x double>, i32) nounwind readnone
declare i64 @llvm.x86.avx512.vcvtsd2usi64(<2 x double>, i32) nounwind readnone
declare i64 @llvm.x86.avx512.cvttsd2usi64(<2 x double>, i32) nounwind readnone
