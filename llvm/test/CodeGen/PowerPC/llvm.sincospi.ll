; RUN: not llc -mtriple=powerpc64le-gnu-linux -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: no libcall available for fsincospi
define { half, half } @test_sincospi_f16(half %a) #0 {
  %result = call { half, half } @llvm.sincospi.f16(half %a)
  ret { half, half } %result
}

; CHECK: error: no libcall available for fsincospi
define { float, float } @test_sincospi_f32(float %a) #0 {
  %result = call { float, float } @llvm.sincospi.f32(float %a)
  ret { float, float } %result
}

; CHECK: error: no libcall available for fsincospi
define { double, double } @test_sincospi_f64(double %a) #0 {
  %result = call { double, double } @llvm.sincospi.f64(double %a)
  ret { double, double } %result
}

attributes #0 = { nounwind }
