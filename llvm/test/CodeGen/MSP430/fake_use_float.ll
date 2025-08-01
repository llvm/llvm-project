; RUN: llc < %s | FileCheck %s
target triple = "msp430"

; CHECK:       mov	#19923, r12
; CHECK:       mov	#4194, r12
; CHECK:       mov	#25688, r12
; CHECK:       mov	#-16245, r12
; CHECK:       ret
define void @test-double() {
entry:
  call void (...) @llvm.fake.use(double -8.765430e+02)
  ret void
}

; CHECK:       call	#__mspabi_addd
; CHECK:       ret
define void @test-double2(double %0) {
entry:
  %1 = fadd double %0, %0
  call void (...) @llvm.fake.use(double %1)
  ret void
}

; CHECK:       call	#__mspabi_addf
; CHECK:       ret
define void @test-float(float %0) {
entry:
  %1 = fadd float %0, %0
  call void (...) @llvm.fake.use(float %1)
  ret void
}
