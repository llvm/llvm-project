; XFAIL: *
; RUN: llc -O3 < %s -mtriple=x86_64-apple-darwin -mattr=+avx512bw | FileCheck %s

define double @foo(i32** nocapture readonly) #0 {
  %2 = load i64, i64* undef, align 8
  %3 = and i64 %2, 1
  %4 = icmp eq i64 %3, 0
  %5 = sitofp i64 %2 to double
  %6 = select i1 %4, double 1.000000e+00, double %5
  ret double %6
}
