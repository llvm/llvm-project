; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00000044

; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Double-precision floating point
; CHECK-NEXT: ;       Double-precision extensions for 11.1
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;
; CHECK-NEXT: ; Shader Flags for Module Functions

;CHECK: ; Function top_level : 0x00000044
define double @top_level() #0 {
  %r = call double @test_uitofp_i64(i64 5)
  ret double %r
}


; CHECK: ; Function test_fdiv_double : 0x00000044
define double @test_fdiv_double(double %a, double %b) #0 {
  %res = fdiv double %a, %b
  ret double %res
}

; CHECK: ; Function test_uitofp_i64 : 0x00000044
define double @test_uitofp_i64(i64 %a) #0 {
  %r = uitofp i64 %a to double
  ret double %r
}

; CHECK: ; Function test_sitofp_i64 : 0x00000044
define double @test_sitofp_i64(i64 %a) #0 {
  %r = sitofp i64 %a to double
  ret double %r
}

; CHECK: ; Function test_fptoui_i32 : 0x00000044
define i32 @test_fptoui_i32(double %a) #0 {
  %r = fptoui double %a to i32
  ret i32 %r
}

; CHECK: ; Function test_fptosi_i64 : 0x00000044
define i64 @test_fptosi_i64(double %a) #0 {
  %r = fptosi double %a to i64
  ret i64 %r
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
