; RUN: opt -passes=early-cse -S < %s | FileCheck %s

@d8 = private constant [8 x double] [double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0, double 8.0], align 64
@d16 = private constant [16 x double] [double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0, double 8.0, double 9.0, double 10.0, double 11.0, double 12.0, double 13.0, double 14.0, double 15.0, double 16.0], align 128
@w32 = private constant [32 x i16] [
  i16 42, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7,
  i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15,
  i16 16, i16 17, i16 18, i16 19, i16 20, i16 21, i16 22, i16 23,
  i16 24, i16 25, i16 26, i16 27, i16 28, i16 29, i16 30, i16 31
], align 2
@str64 = private constant [64 x i8] c"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", align 1

define <8 x double> @fold_8xdouble_load() {
  ; CHECK-LABEL: @fold_8xdouble_load(
  ; CHECK: ret <8 x double> <double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00, double 5.000000e+00, double 6.000000e+00, double 7.000000e+00, double 8.000000e+00>
  %v = load <8 x double>, ptr @d8, align 64
  ret <8 x double> %v
}

define double @fold_16xdouble_load() {
  ; CHECK-LABEL: @fold_16xdouble_load(
  ; CHECK: ret double 2.000000e+00
  %v = load <16 x double>, ptr @d16, align 128
  %e = extractelement <16 x double> %v, i32 1
  ret double %e
}

define i16 @fold_array_to_i512() {
  ; CHECK-LABEL: @fold_array_to_i512(
  ; CHECK: ret i16 42
  %v = load i512, ptr @w32, align 2
  %e = trunc i512 %v to i16
  ret i16 %e
}

define i16 @fold_array_to_vec() {
  ; CHECK-LABEL: @fold_array_to_vec(
  ; CHECK: ret i16 42
  %v = load <32 x i16>, ptr @w32, align 2
  %e = extractelement <32 x i16> %v, i32 0
  ret i16 %e
}

define i8 @not_fold_string_to_i512() {
  ; CHECK-LABEL: @not_fold_string_to_i512(
  ; CHECK: %v = load i512, ptr @str64, align 1
  ; CHECK: %e = trunc i512 %v to i8
  ; CHECK: ret i8 %e
  %v = load i512, ptr @str64, align 1
  %e = trunc i512 %v to i8
  ret i8 %e
}

define i8 @not_fold_string_to_vec() {
  ; CHECK-LABEL: @not_fold_string_to_vec(
  ; CHECK: %v = load <64 x i8>, ptr @str64, align 1
  ; CHECK: %e = extractelement <64 x i8> %v, i32 0
  ; CHECK: ret i8 %e
  %v = load <64 x i8>, ptr @str64, align 1
  %e = extractelement <64 x i8> %v, i32 0
  ret i8 %e
}
