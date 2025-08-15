; RUN: not llc -mtriple=avr -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: do not know how to soften fsincos
define { double, double } @test_sincos_f64(double %a) #0 {
  %result = call { double, double } @llvm.sincos.f64(double %a)
  ret { double, double } %result
}

attributes #0 = { nounwind }
