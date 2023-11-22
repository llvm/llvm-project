; RUN: opt -passes=verify -S < %s 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: opt -passes=verify -S < %s 2>&1 | FileCheck --check-prefix=CHECK2 %s
; RUN: sed -e s/.T3:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK3 %s
; RUN: sed -e s/.T4:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK4 %s

; Check that a double-valued complex fmul is accepted, and attributes are
; correct.
; CHECK1: declare <2 x double> @llvm.experimental.complex.fmul.v2f64(<2 x double>, <2 x double>) #[[ATTR:[0-9]+]]
; CHECK1:  attributes #[[ATTR]] = { nocallback nofree nosync nounwind willreturn memory(none) }
declare <2 x double> @llvm.experimental.complex.fmul.v2f64(<2 x double>, <2 x double>)
define <2 x double> @t1(<2 x double> %a, <2 x double> %b) {
  %res = call <2 x double> @llvm.experimental.complex.fmul.v2f64(<2 x double> %a, <2 x double> %b)
  ret <2 x double> %res
}

; Test that vector complex values are supported.
; CHECK2: declare <4 x double> @llvm.experimental.complex.fmul.v4f64(<4 x double>, <4 x double>) #[[ATTR:[0-9]+]]
; CHECK2:  attributes #[[ATTR]] = { nocallback nofree nosync nounwind willreturn memory(none) }
declare <4 x double> @llvm.experimental.complex.fmul.v4f64(<4 x double>, <4 x double>)
define <4 x double> @t2(<4 x double> %a, <4 x double> %b) {
  %res = call <4 x double> @llvm.experimental.complex.fmul.v4f64(<4 x double> %a, <4 x double> %b)
  ret <4 x double> %res
}

; Test that odd-length vectors are not supported.
; CHECK3: complex intrinsic must use an even-length vector of floating-point types
;T3: declare <3 x double> @llvm.experimental.complex.fmul.v3f64(<3 x double>, <3 x double>)
;T3: define <3 x double> @t3(<3 x double> %a, <3 x double> %b) {
;T3:   %res = call <3 x double> @llvm.experimental.complex.fmul.v3f64(<3 x double> %a, <3 x double> %b)
;T3:   ret <3 x double> %res
;T3: }

; Test that non-floating point complex types are not supported.
; CHECK4: complex intrinsic must use an even-length vector of floating-point types
;T4: declare <2 x i64> @llvm.experimental.complex.fmul.v2i64(<2 x i64>, <2 x i64>)
;T4: define <2 x i64> @t4(<2 x i64> %a, <2 x i64> %b) {
;T4:   %res = call <2 x i64> @llvm.experimental.complex.fmul.v2i64(<2 x i64> %a, <2 x i64> %b)
;T4:   ret <2 x i64> %res
;T4: }
