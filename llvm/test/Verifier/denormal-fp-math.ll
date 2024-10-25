; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck --implicit-check-not="invalid value" %s

define float @test_denormal_fp_math_valid() "denormal-fp-math"="ieee,ieee" {
  ret float 1.0
}

; CHECK: invalid value for 'denormal-fp-math' attribute: foo,ieee
define float @test_denormal_fp_math_invalid1() "denormal-fp-math"="foo,ieee" {
  ret float 1.0
}

; CHECK: invalid value for 'denormal-fp-math' attribute: ieee,ieee,ieee
define float @test_denormal_fp_math_invalid2() "denormal-fp-math"="ieee,ieee,ieee" {
  ret float 1.0
}

; CHECK: invalid value for 'denormal-fp-math-f32' attribute: foo,ieee
define float @test_denormal_fp_math_f32_invalid() "denormal-fp-math-f32"="foo,ieee" {
  ret float 1.0
}
