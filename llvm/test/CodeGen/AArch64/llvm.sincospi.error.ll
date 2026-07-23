; RUN: not llc -mtriple=aarch64-gnu-linux -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: no libcall available for fsincospi
define { float, float } @test_sincospi_f32(float %a) {
  %result = call { float, float } @llvm.sincospi.f32(float %a)
  ret { float, float } %result
}

; CHECK: error: no libcall available for fsincospi
define { double, double } @test_sincospi_f64(double %a) {
  %result = call { double, double } @llvm.sincospi.f64(double %a)
  ret { double, double } %result
}
