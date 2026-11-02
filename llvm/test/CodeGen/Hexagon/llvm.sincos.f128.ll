; RUN: not llc -mtriple=hexagon-unknown-linux-gnu -filetype=null %s 2>&1 | FileCheck %s

; Hexagon long double is IEEE double, so there is no fp128 sincosl.

; CHECK: error: do not know how to soften fsincos
define { fp128, fp128 } @test_sincos_f128(fp128 %a) {
  %result = call { fp128, fp128 } @llvm.sincos.f128(fp128 %a)
  ret { fp128, fp128 } %result
}
