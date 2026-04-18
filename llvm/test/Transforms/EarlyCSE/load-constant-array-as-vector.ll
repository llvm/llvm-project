; RUN: opt -passes=early-cse -S < %s | FileCheck %s

@d16 = private constant [16 x double] [double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0, double 8.0, double 9.0, double 10.0, double 11.0, double 12.0, double 13.0, double 14.0, double 15.0, double 16.0], align 128

define double @fold_16xdouble_load() {
  ; CHECK: ret double 2.000000e+00
  %v = load <16 x double>, ptr @d16, align 128
  %e = extractelement <16 x double> %v, i32 1
  ret double %e
}
