; RUN: opt < %s -slp-vectorizer -mtriple=ve-linux -S | FileCheck %s -check-prefix=VE 
; RUN: opt < %s -slp-vectorizer -mtriple=x86_64-pc_linux -mcpu=core-avx2 -S | FileCheck %s -check-prefix=SSE

; Make sure SLP does not trigger for VE on an appealing set of combinable loads
; and stores that vectorizes for x86 SSE.

; TODO: Remove this test once VE vector isel is deemed stable.

; VE-NOT: x double
; SSE: x double

define void @foo(double* noalias %A0p, double* noalias %B0p) {
entry:
  %A1p = getelementptr inbounds double, double* %A0p, i64 1
  %A2p = getelementptr inbounds double, double* %A0p, i64 2
  %A3p = getelementptr inbounds double, double* %A0p, i64 3
  %A4p = getelementptr inbounds double, double* %A0p, i64 4
  %A5p = getelementptr inbounds double, double* %A0p, i64 5
  %A6p = getelementptr inbounds double, double* %A0p, i64 6
  %A7p = getelementptr inbounds double, double* %A0p, i64 7
  %A8p = getelementptr inbounds double, double* %A0p, i64 8
  %A9p = getelementptr inbounds double, double* %A0p, i64 9
  %A10p = getelementptr inbounds double, double* %A0p, i64 10
  %A11p = getelementptr inbounds double, double* %A0p, i64 11
  %A12p = getelementptr inbounds double, double* %A0p, i64 12
  %A13p = getelementptr inbounds double, double* %A0p, i64 13
  %A14p = getelementptr inbounds double, double* %A0p, i64 14
  %A15p = getelementptr inbounds double, double* %A0p, i64 15
  %A0 = load double, double* %A0p, align 8
  %A1 = load double, double* %A1p, align 8
  %A2 = load double, double* %A2p, align 8
  %A3 = load double, double* %A3p, align 8
  %A4 = load double, double* %A4p, align 8
  %A5 = load double, double* %A5p, align 8
  %A6 = load double, double* %A6p, align 8
  %A7 = load double, double* %A7p, align 8
  %A8 = load double, double* %A8p, align 8
  %A9 = load double, double* %A9p, align 8
  %A10 = load double, double* %A10p, align 8
  %A11 = load double, double* %A11p, align 8
  %A12 = load double, double* %A12p, align 8
  %A13 = load double, double* %A13p, align 8
  %A14 = load double, double* %A14p, align 8
  %A15 = load double, double* %A15p, align 8
  %B1p = getelementptr inbounds double, double* %B0p, i64 1
  %B2p = getelementptr inbounds double, double* %B0p, i64 2
  %B3p = getelementptr inbounds double, double* %B0p, i64 3
  %B4p = getelementptr inbounds double, double* %B0p, i64 4
  %B5p = getelementptr inbounds double, double* %B0p, i64 5
  %B6p = getelementptr inbounds double, double* %B0p, i64 6
  %B7p = getelementptr inbounds double, double* %B0p, i64 7
  %B8p = getelementptr inbounds double, double* %B0p, i64 8
  %B9p = getelementptr inbounds double, double* %B0p, i64 9
  %B10p = getelementptr inbounds double, double* %B0p, i64 10
  %B11p = getelementptr inbounds double, double* %B0p, i64 11
  %B12p = getelementptr inbounds double, double* %B0p, i64 12
  %B13p = getelementptr inbounds double, double* %B0p, i64 13
  %B14p = getelementptr inbounds double, double* %B0p, i64 14
  %B15p = getelementptr inbounds double, double* %B0p, i64 15
  store double %A0, double* %B0p, align 8
  store double %A1, double* %B1p, align 8
  store double %A2, double* %B2p, align 8
  store double %A3, double* %B3p, align 8
  store double %A4, double* %B4p, align 8
  store double %A5, double* %B5p, align 8
  store double %A6, double* %B6p, align 8
  store double %A7, double* %B7p, align 8
  store double %A8, double* %B8p, align 8
  store double %A9, double* %B9p, align 8
  store double %A10, double* %B10p, align 8
  store double %A11, double* %B11p, align 8
  store double %A12, double* %B12p, align 8
  store double %A13, double* %B13p, align 8
  store double %A14, double* %B14p, align 8
  store double %A15, double* %B15p, align 8
  ret void
}
