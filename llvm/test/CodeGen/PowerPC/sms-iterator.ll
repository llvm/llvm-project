; REQUIRES: asserts
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs\
; RUN:       -mcpu=pwr9 --ppc-enable-pipeliner -debug-only=pipeliner 2>&1 \
; RUN:       >/dev/null | FileCheck %s

%0 = type { i32, [16 x double] }

; CHECK: MII = 3 MAX_II = 13 (rec=3, res=2)

define dso_local fastcc double @_ZN3povL9polysolveEiPdS0_() unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  br i1 undef, label %2, label %1

2:                                                ; preds = %1
  br i1 undef, label %14, label %3

3:                                                ; preds = %3, %2
  %4 = phi i64 [ %7, %3 ], [ undef, %2 ]
  %5 = phi double [ %11, %3 ], [ undef, %2 ]
  %6 = phi i64 [ %12, %3 ], [ undef, %2 ]
  %7 = add nsw i64 %4, -1
  %8 = fmul fast double %5, 1.000000e+07
  %9 = getelementptr inbounds %0, ptr null, i64 1, i32 1, i64 %7
  %10 = load double, ptr %9, align 8
  %11 = fadd fast double %10, %8
  %12 = add i64 %6, -1
  %13 = icmp eq i64 %12, 0
  br i1 %13, label %14, label %3

14:                                               ; preds = %3, %2
  %15 = phi double [ undef, %2 ], [ %11, %3 ]
  %16 = fmul fast double %15, undef
  ret double %16
}
