; RUN: opt < %s -constprop -S | FileCheck %s
; RUN: opt < %s -constprop -disable-simplify-libcalls -S | FileCheck %s --check-prefix=FNOBUILTIN
; REQUIRES: x86

define i1 @test_sse_cvts_exact() nounwind readnone {
; CHECK-LABEL: @test_sse_cvts_exact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.sse.cvtss2si(<4 x float> <float 3.0, float undef, float undef, float undef>) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvtss2si64(<4 x float> <float 3.0, float undef, float undef, float undef>) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> <double 7.0, double undef>) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> <double 7.0, double undef>) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 10
  %cmp13 = icmp eq i64 %sum13, 10
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

; Inexact values should not fold as they are dependent on rounding mode
define i1 @test_sse_cvts_inexact() nounwind readnone {
; CHECK-LABEL: @test_sse_cvts_inexact(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %i0 = tail call i32 @llvm.x86.sse.cvtss2si(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvtss2si64(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> <double 1.75, double undef>) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> <double 1.75, double undef>) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 4
  %cmp13 = icmp eq i64 %sum13, 4
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

; FLT_MAX/DBL_MAX should not fold
define i1 @test_sse_cvts_max() nounwind readnone {
; CHECK-LABEL: @test_sse_cvts_max(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095039, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405311, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.sse.cvtss2si(<4 x float> %fm) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %fm) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %dm) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %dm) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; INF should not fold
define i1 @test_sse_cvts_inf() nounwind readnone {
; CHECK-LABEL: @test_sse_cvts_inf(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095040, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405312, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.sse.cvtss2si(<4 x float> %fm) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %fm) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %dm) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %dm) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; NAN should not fold
define i1 @test_sse_cvts_nan() nounwind readnone {
; CHECK-LABEL: @test_sse_cvts_nan(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2143289344, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9221120237041090560, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.sse.cvtss2si(<4 x float> %fm) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %fm) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %dm) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %dm) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

define i1 @test_sse_cvtts_exact() nounwind readnone {
; CHECK-LABEL: @test_sse_cvtts_exact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.sse.cvttss2si(<4 x float> <float 3.0, float undef, float undef, float undef>) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvttss2si64(<4 x float> <float 3.0, float undef, float undef, float undef>) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> <double 7.0, double undef>) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> <double 7.0, double undef>) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 10
  %cmp13 = icmp eq i64 %sum13, 10
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

define i1 @test_sse_cvtts_inexact() nounwind readnone {
; CHECK-LABEL: @test_sse_cvtts_inexact(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.sse.cvttss2si(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvttss2si64(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> <double 1.75, double undef>) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> <double 1.75, double undef>) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %cmp02 = icmp eq i32 %sum02, 2
  %cmp13 = icmp eq i64 %sum13, 2
  %b = and i1 %cmp02, %cmp13
  ret i1 %b
}

; FLT_MAX/DBL_MAX should not fold
define i1 @test_sse_cvtts_max() nounwind readnone {
; CHECK-LABEL: @test_sse_cvtts_max(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095039, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405311, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.sse.cvttss2si(<4 x float> %fm) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %fm) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %dm) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %dm) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; INF should not fold
define i1 @test_sse_cvtts_inf() nounwind readnone {
; CHECK-LABEL: @test_sse_cvtts_inf(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2139095040, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9218868437227405312, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.sse.cvttss2si(<4 x float> %fm) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %fm) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %dm) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %dm) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

; NAN should not fold
define i1 @test_sse_cvtts_nan() nounwind readnone {
; CHECK-LABEL: @test_sse_cvtts_nan(
; CHECK: call
; CHECK: call
; CHECK: call
; CHECK: call
entry:
  %fm = bitcast <4 x i32> <i32 2143289344, i32 undef, i32 undef, i32 undef> to <4 x float>
  %dm = bitcast <2 x i64> <i64 9221120237041090560, i64 undef> to <2 x double>
  %i0 = tail call i32 @llvm.x86.sse.cvttss2si(<4 x float> %fm) nounwind
  %i1 = tail call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %fm) nounwind
  %i2 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %dm) nounwind
  %i3 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %dm) nounwind
  %sum02 = add i32 %i0, %i2
  %sum13 = add i64 %i1, %i3
  %sum02.sext = sext i32 %sum02 to i64
  %b = icmp eq i64 %sum02.sext, %sum13
  ret i1 %b
}

declare i32 @llvm.x86.sse.cvtss2si(<4 x float>) nounwind readnone
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone
declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>) nounwind readnone
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>) nounwind readnone
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone
