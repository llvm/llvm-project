; RUN: llc < %s -mtriple=i686-pc-elfiamcu | FileCheck %s

; CHECK-LABEL: test_lib_args:
; CHECK: movl %edx, %eax
; CHECK: calll __fixsfsi
define i32 @test_lib_args(float inreg %a, float inreg %b) #0 {
  %ret = fptosi float %b to i32
  ret i32 %ret
}

attributes #0 = { nounwind "use-soft-float"="true"}
