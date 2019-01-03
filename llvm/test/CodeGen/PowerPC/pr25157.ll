; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s -verify-machineinstrs | FileCheck %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s -verify-machineinstrs | FileCheck \
; RUN:   --check-prefix=CHECK-P9 %s

; Verify correct generation of an lxsspx rather than an invalid optimization
; to lxvdsx.  Bugpoint-reduced test from Eric Schweitz.

%struct.BSS38.51.4488.9911.14348.16813.20264.24701.28152.31603.35054.39491.44914.45407.46393.46886.47872.49351.49844.50830.51323.52309.53295.53788.54281.55267.55760.59211.61625 = type <{ [28 x i8] }>
%struct_main1_2_.491.4928.10351.14788.17253.20704.25141.28592.32043.35494.39931.45354.45847.46833.47326.48312.49791.50284.51270.51763.52749.53735.54228.54721.55707.56200.59651.61626 = type <{ [64 x i8] }>

@.BSS38 = external global %struct.BSS38.51.4488.9911.14348.16813.20264.24701.28152.31603.35054.39491.44914.45407.46393.46886.47872.49351.49844.50830.51323.52309.53295.53788.54281.55267.55760.59211.61625, align 32
@_main1_2_ = external global %struct_main1_2_.491.4928.10351.14788.17253.20704.25141.28592.32043.35494.39931.45354.45847.46833.47326.48312.49791.50284.51270.51763.52749.53735.54228.54721.55707.56200.59651.61626, section ".comm", align 16

define void @aercalc_() {
L.entry:
  br i1 undef, label %L.LB38_2426, label %L.LB38_2911

L.LB38_2911:
  br i1 undef, label %L.LB38_2140, label %L.LB38_2640

L.LB38_2640:
  unreachable

L.LB38_2426:
  br i1 undef, label %L.LB38_2438, label %L.LB38_2920

L.LB38_2920:
  br i1 undef, label %L.LB38_2438, label %L.LB38_2921

L.LB38_2921:
  br label %L.LB38_2140

L.LB38_2140:
  ret void

L.LB38_2438:
  br i1 undef, label %L.LB38_2451, label %L.LB38_2935

L.LB38_2935:
  br i1 undef, label %L.LB38_2451, label %L.LB38_2936

L.LB38_2936:
  unreachable

L.LB38_2451:
  br i1 undef, label %L.LB38_2452, label %L.LB38_2937

L.LB38_2937:
  unreachable

L.LB38_2452:
  %0 = load float, float* bitcast (i8* getelementptr inbounds (%struct.BSS38.51.4488.9911.14348.16813.20264.24701.28152.31603.35054.39491.44914.45407.46393.46886.47872.49351.49844.50830.51323.52309.53295.53788.54281.55267.55760.59211.61625, %struct.BSS38.51.4488.9911.14348.16813.20264.24701.28152.31603.35054.39491.44914.45407.46393.46886.47872.49351.49844.50830.51323.52309.53295.53788.54281.55267.55760.59211.61625* @.BSS38, i64 0, i32 0, i64 16) to float*), align 16
  %1 = fpext float %0 to double
  %2 = insertelement <2 x double> undef, double %1, i32 1
  store <2 x double> %2, <2 x double>* bitcast (i8* getelementptr inbounds (%struct_main1_2_.491.4928.10351.14788.17253.20704.25141.28592.32043.35494.39931.45354.45847.46833.47326.48312.49791.50284.51270.51763.52749.53735.54228.54721.55707.56200.59651.61626, %struct_main1_2_.491.4928.10351.14788.17253.20704.25141.28592.32043.35494.39931.45354.45847.46833.47326.48312.49791.50284.51270.51763.52749.53735.54228.54721.55707.56200.59651.61626* @_main1_2_, i64 0, i32 0, i64 32) to <2 x double>*), align 16
  unreachable
}

; CHECK-LABEL: @aercalc_
; CHECK: lfs
; CHECK-P9-LABEL: @aercalc_
; CHECK-P9: lfs
