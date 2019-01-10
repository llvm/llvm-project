; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mattr=-fullfp16 \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOFP16
; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mattr=+fullfp16 \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FP16

;;; Half vectors

%v4f16 = type <4 x half>

define %v4f16 @test_v4f16.sqrt(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.sqrt:
  ; CHECK-NOFP16-COUNT-4: fsqrt s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fsqrt.4h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.sqrt.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.powi(%v4f16 %a, i32 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.powi:
  ; CHECK-COUNT-4: bl __powi
  %1 = call %v4f16 @llvm.powi.v4f16(%v4f16 %a, i32 %b)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.sin(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.sin:
  ; CHECK-COUNT-4: bl sinf
  %1 = call %v4f16 @llvm.sin.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.cos(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.cos:
  ; CHECK-COUNT-4: bl cosf
  %1 = call %v4f16 @llvm.cos.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.pow(%v4f16 %a, %v4f16 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.pow:
  ; CHECK-COUNT-4: bl pow
  %1 = call %v4f16 @llvm.pow.v4f16(%v4f16 %a, %v4f16 %b)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.exp(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.exp:
  ; CHECK-COUNT-4: bl exp
  %1 = call %v4f16 @llvm.exp.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.exp2(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.exp2:
  ; CHECK-COUNT-4: bl exp2
  %1 = call %v4f16 @llvm.exp2.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.log(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.log:
  ; CHECK-COUNT-4: bl log
  %1 = call %v4f16 @llvm.log.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.log10(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.log10:
  ; CHECK-COUNT-4: bl log10
  %1 = call %v4f16 @llvm.log10.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.log2(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.log2:
  ; CHECK-COUNT-4: bl log2
  %1 = call %v4f16 @llvm.log2.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.fma(%v4f16 %a, %v4f16 %b, %v4f16 %c) {
  ; CHECK-LABEL:          test_v4f16.fma:
  ; CHECK-NOFP16-COUNT-4: fmadd s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fmla.4h
  %1 = call %v4f16 @llvm.fma.v4f16(%v4f16 %a, %v4f16 %b, %v4f16 %c)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.fabs(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.fabs:
  ; CHECK-NOFP16-COUNT-4: fabs s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fabs.4h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.fabs.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.floor(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.floor:
  ; CHECK-NOFP16-COUNT-4: frintm s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintm.4h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.floor.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.ceil(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.ceil:
  ; CHECK-NOFP16-COUNT-4: frintp s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintp.4h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.ceil.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.trunc(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.trunc:
  ; CHECK-NOFP16-COUNT-4: frintz s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintz.4h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.trunc.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.rint(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.rint:
  ; CHECK-NOFP16-COUNT-4: frintx s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintx.4h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.rint.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.nearbyint(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.nearbyint:
  ; CHECK-NOFP16-COUNT-4: frinti s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frinti.4h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.nearbyint.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

declare %v4f16 @llvm.sqrt.v4f16(%v4f16) #0
declare %v4f16 @llvm.powi.v4f16(%v4f16, i32) #0
declare %v4f16 @llvm.sin.v4f16(%v4f16) #0
declare %v4f16 @llvm.cos.v4f16(%v4f16) #0
declare %v4f16 @llvm.pow.v4f16(%v4f16, %v4f16) #0
declare %v4f16 @llvm.exp.v4f16(%v4f16) #0
declare %v4f16 @llvm.exp2.v4f16(%v4f16) #0
declare %v4f16 @llvm.log.v4f16(%v4f16) #0
declare %v4f16 @llvm.log10.v4f16(%v4f16) #0
declare %v4f16 @llvm.log2.v4f16(%v4f16) #0
declare %v4f16 @llvm.fma.v4f16(%v4f16, %v4f16, %v4f16) #0
declare %v4f16 @llvm.fabs.v4f16(%v4f16) #0
declare %v4f16 @llvm.floor.v4f16(%v4f16) #0
declare %v4f16 @llvm.ceil.v4f16(%v4f16) #0
declare %v4f16 @llvm.trunc.v4f16(%v4f16) #0
declare %v4f16 @llvm.rint.v4f16(%v4f16) #0
declare %v4f16 @llvm.nearbyint.v4f16(%v4f16) #0

;;;

%v8f16 = type <8 x half>

define %v8f16 @test_v8f16.sqrt(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.sqrt:
  ; CHECK-NOFP16-COUNT-8: fsqrt s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fsqrt.8h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.sqrt.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.powi(%v8f16 %a, i32 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.powi:
  ; CHECK-COUNT-8: bl __powi
  %1 = call %v8f16 @llvm.powi.v8f16(%v8f16 %a, i32 %b)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.sin(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.sin:
  ; CHECK-COUNT-8: bl sinf
  %1 = call %v8f16 @llvm.sin.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.cos(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.cos:
  ; CHECK-COUNT-8: bl cosf
  %1 = call %v8f16 @llvm.cos.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.pow(%v8f16 %a, %v8f16 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.pow:
  ; CHECK-COUNT-8: bl pow
  %1 = call %v8f16 @llvm.pow.v8f16(%v8f16 %a, %v8f16 %b)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.exp(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.exp:
  ; CHECK-COUNT-8: bl exp
  %1 = call %v8f16 @llvm.exp.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.exp2(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.exp2:
  ; CHECK-COUNT-8: bl exp2
  %1 = call %v8f16 @llvm.exp2.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.log(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.log:
  ; CHECK-COUNT-8: bl log
  %1 = call %v8f16 @llvm.log.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.log10(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.log10:
  ; CHECK-COUNT-8: bl log10
  %1 = call %v8f16 @llvm.log10.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.log2(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.log2:
  ; CHECK-COUNT-8: bl log2
  %1 = call %v8f16 @llvm.log2.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.fma(%v8f16 %a, %v8f16 %b, %v8f16 %c) {
  ; CHECK-LABEL:          test_v8f16.fma:
  ; CHECK-NOFP16-COUNT-8: fmadd s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fmla.8h
  %1 = call %v8f16 @llvm.fma.v8f16(%v8f16 %a, %v8f16 %b, %v8f16 %c)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.fabs(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.fabs:
  ; CHECK-NOFP16-COUNT-8: fabs s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fabs.8h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.fabs.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.floor(%v8f16 %a) {
  ; CHECK-LABEL:     		  test_v8f16.floor:
  ; CHECK-NOFP16-COUNT-8: frintm s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintm.8h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.floor.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.ceil(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.ceil:
  ; CHECK-NOFP16-COUNT-8: frintp s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintp.8h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.ceil.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.trunc(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.trunc:
  ; CHECK-NOFP16-COUNT-8: frintz s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintz.8h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.trunc.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.rint(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.rint:
  ; CHECK-NOFP16-COUNT-8: frintx s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintx.8h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.rint.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.nearbyint(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.nearbyint:
  ; CHECK-NOFP16-COUNT-8: frinti s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frinti.8h
  ; CHECK-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.nearbyint.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

declare %v8f16 @llvm.sqrt.v8f16(%v8f16) #0
declare %v8f16 @llvm.powi.v8f16(%v8f16, i32) #0
declare %v8f16 @llvm.sin.v8f16(%v8f16) #0
declare %v8f16 @llvm.cos.v8f16(%v8f16) #0
declare %v8f16 @llvm.pow.v8f16(%v8f16, %v8f16) #0
declare %v8f16 @llvm.exp.v8f16(%v8f16) #0
declare %v8f16 @llvm.exp2.v8f16(%v8f16) #0
declare %v8f16 @llvm.log.v8f16(%v8f16) #0
declare %v8f16 @llvm.log10.v8f16(%v8f16) #0
declare %v8f16 @llvm.log2.v8f16(%v8f16) #0
declare %v8f16 @llvm.fma.v8f16(%v8f16, %v8f16, %v8f16) #0
declare %v8f16 @llvm.fabs.v8f16(%v8f16) #0
declare %v8f16 @llvm.floor.v8f16(%v8f16) #0
declare %v8f16 @llvm.ceil.v8f16(%v8f16) #0
declare %v8f16 @llvm.trunc.v8f16(%v8f16) #0
declare %v8f16 @llvm.rint.v8f16(%v8f16) #0
declare %v8f16 @llvm.nearbyint.v8f16(%v8f16) #0

;;; Float vectors

%v2f32 = type <2 x float>

; CHECK-LABEL: test_v2f32.sqrt:
define %v2f32 @test_v2f32.sqrt(%v2f32 %a) {
  ; CHECK: fsqrt.2s
  %1 = call %v2f32 @llvm.sqrt.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.powi:
define %v2f32 @test_v2f32.powi(%v2f32 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v2f32 @llvm.powi.v2f32(%v2f32 %a, i32 %b)
  ret %v2f32 %1
}
; CHECK: test_v2f32.sin:
define %v2f32 @test_v2f32.sin(%v2f32 %a) {
  ; CHECK: sin
  %1 = call %v2f32 @llvm.sin.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.cos:
define %v2f32 @test_v2f32.cos(%v2f32 %a) {
  ; CHECK: cos
  %1 = call %v2f32 @llvm.cos.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.pow:
define %v2f32 @test_v2f32.pow(%v2f32 %a, %v2f32 %b) {
  ; CHECK: pow
  %1 = call %v2f32 @llvm.pow.v2f32(%v2f32 %a, %v2f32 %b)
  ret %v2f32 %1
}
; CHECK: test_v2f32.exp:
define %v2f32 @test_v2f32.exp(%v2f32 %a) {
  ; CHECK: exp
  %1 = call %v2f32 @llvm.exp.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.exp2:
define %v2f32 @test_v2f32.exp2(%v2f32 %a) {
  ; CHECK: exp
  %1 = call %v2f32 @llvm.exp2.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.log:
define %v2f32 @test_v2f32.log(%v2f32 %a) {
  ; CHECK: log
  %1 = call %v2f32 @llvm.log.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.log10:
define %v2f32 @test_v2f32.log10(%v2f32 %a) {
  ; CHECK: log
  %1 = call %v2f32 @llvm.log10.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.log2:
define %v2f32 @test_v2f32.log2(%v2f32 %a) {
  ; CHECK: log
  %1 = call %v2f32 @llvm.log2.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.fma:
define %v2f32 @test_v2f32.fma(%v2f32 %a, %v2f32 %b, %v2f32 %c) {
  ; CHECK: fmla.2s
  %1 = call %v2f32 @llvm.fma.v2f32(%v2f32 %a, %v2f32 %b, %v2f32 %c)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.fabs:
define %v2f32 @test_v2f32.fabs(%v2f32 %a) {
  ; CHECK: fabs.2s
  %1 = call %v2f32 @llvm.fabs.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.floor:
define %v2f32 @test_v2f32.floor(%v2f32 %a) {
  ; CHECK: frintm.2s
  %1 = call %v2f32 @llvm.floor.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.ceil:
define %v2f32 @test_v2f32.ceil(%v2f32 %a) {
  ; CHECK: frintp.2s
  %1 = call %v2f32 @llvm.ceil.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.trunc:
define %v2f32 @test_v2f32.trunc(%v2f32 %a) {
  ; CHECK: frintz.2s
  %1 = call %v2f32 @llvm.trunc.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.rint:
define %v2f32 @test_v2f32.rint(%v2f32 %a) {
  ; CHECK: frintx.2s
  %1 = call %v2f32 @llvm.rint.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.nearbyint:
define %v2f32 @test_v2f32.nearbyint(%v2f32 %a) {
  ; CHECK: frinti.2s
  %1 = call %v2f32 @llvm.nearbyint.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

declare %v2f32 @llvm.sqrt.v2f32(%v2f32) #0
declare %v2f32 @llvm.powi.v2f32(%v2f32, i32) #0
declare %v2f32 @llvm.sin.v2f32(%v2f32) #0
declare %v2f32 @llvm.cos.v2f32(%v2f32) #0
declare %v2f32 @llvm.pow.v2f32(%v2f32, %v2f32) #0
declare %v2f32 @llvm.exp.v2f32(%v2f32) #0
declare %v2f32 @llvm.exp2.v2f32(%v2f32) #0
declare %v2f32 @llvm.log.v2f32(%v2f32) #0
declare %v2f32 @llvm.log10.v2f32(%v2f32) #0
declare %v2f32 @llvm.log2.v2f32(%v2f32) #0
declare %v2f32 @llvm.fma.v2f32(%v2f32, %v2f32, %v2f32) #0
declare %v2f32 @llvm.fabs.v2f32(%v2f32) #0
declare %v2f32 @llvm.floor.v2f32(%v2f32) #0
declare %v2f32 @llvm.ceil.v2f32(%v2f32) #0
declare %v2f32 @llvm.trunc.v2f32(%v2f32) #0
declare %v2f32 @llvm.rint.v2f32(%v2f32) #0
declare %v2f32 @llvm.nearbyint.v2f32(%v2f32) #0

;;;

%v4f32 = type <4 x float>

; CHECK: test_v4f32.sqrt:
define %v4f32 @test_v4f32.sqrt(%v4f32 %a) {
  ; CHECK: fsqrt.4s
  %1 = call %v4f32 @llvm.sqrt.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.powi:
define %v4f32 @test_v4f32.powi(%v4f32 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v4f32 @llvm.powi.v4f32(%v4f32 %a, i32 %b)
  ret %v4f32 %1
}
; CHECK: test_v4f32.sin:
define %v4f32 @test_v4f32.sin(%v4f32 %a) {
  ; CHECK: sin
  %1 = call %v4f32 @llvm.sin.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.cos:
define %v4f32 @test_v4f32.cos(%v4f32 %a) {
  ; CHECK: cos
  %1 = call %v4f32 @llvm.cos.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.pow:
define %v4f32 @test_v4f32.pow(%v4f32 %a, %v4f32 %b) {
  ; CHECK: pow
  %1 = call %v4f32 @llvm.pow.v4f32(%v4f32 %a, %v4f32 %b)
  ret %v4f32 %1
}
; CHECK: test_v4f32.exp:
define %v4f32 @test_v4f32.exp(%v4f32 %a) {
  ; CHECK: exp
  %1 = call %v4f32 @llvm.exp.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.exp2:
define %v4f32 @test_v4f32.exp2(%v4f32 %a) {
  ; CHECK: exp
  %1 = call %v4f32 @llvm.exp2.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.log:
define %v4f32 @test_v4f32.log(%v4f32 %a) {
  ; CHECK: log
  %1 = call %v4f32 @llvm.log.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.log10:
define %v4f32 @test_v4f32.log10(%v4f32 %a) {
  ; CHECK: log
  %1 = call %v4f32 @llvm.log10.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.log2:
define %v4f32 @test_v4f32.log2(%v4f32 %a) {
  ; CHECK: log
  %1 = call %v4f32 @llvm.log2.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.fma:
define %v4f32 @test_v4f32.fma(%v4f32 %a, %v4f32 %b, %v4f32 %c) {
  ; CHECK: fma
  %1 = call %v4f32 @llvm.fma.v4f32(%v4f32 %a, %v4f32 %b, %v4f32 %c)
  ret %v4f32 %1
}
; CHECK: test_v4f32.fabs:
define %v4f32 @test_v4f32.fabs(%v4f32 %a) {
  ; CHECK: fabs
  %1 = call %v4f32 @llvm.fabs.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.floor:
define %v4f32 @test_v4f32.floor(%v4f32 %a) {
  ; CHECK: frintm.4s
  %1 = call %v4f32 @llvm.floor.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.ceil:
define %v4f32 @test_v4f32.ceil(%v4f32 %a) {
  ; CHECK: frintp.4s
  %1 = call %v4f32 @llvm.ceil.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.trunc:
define %v4f32 @test_v4f32.trunc(%v4f32 %a) {
  ; CHECK: frintz.4s
  %1 = call %v4f32 @llvm.trunc.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.rint:
define %v4f32 @test_v4f32.rint(%v4f32 %a) {
  ; CHECK: frintx.4s
  %1 = call %v4f32 @llvm.rint.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.nearbyint:
define %v4f32 @test_v4f32.nearbyint(%v4f32 %a) {
  ; CHECK: frinti.4s
  %1 = call %v4f32 @llvm.nearbyint.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

declare %v4f32 @llvm.sqrt.v4f32(%v4f32) #0
declare %v4f32 @llvm.powi.v4f32(%v4f32, i32) #0
declare %v4f32 @llvm.sin.v4f32(%v4f32) #0
declare %v4f32 @llvm.cos.v4f32(%v4f32) #0
declare %v4f32 @llvm.pow.v4f32(%v4f32, %v4f32) #0
declare %v4f32 @llvm.exp.v4f32(%v4f32) #0
declare %v4f32 @llvm.exp2.v4f32(%v4f32) #0
declare %v4f32 @llvm.log.v4f32(%v4f32) #0
declare %v4f32 @llvm.log10.v4f32(%v4f32) #0
declare %v4f32 @llvm.log2.v4f32(%v4f32) #0
declare %v4f32 @llvm.fma.v4f32(%v4f32, %v4f32, %v4f32) #0
declare %v4f32 @llvm.fabs.v4f32(%v4f32) #0
declare %v4f32 @llvm.floor.v4f32(%v4f32) #0
declare %v4f32 @llvm.ceil.v4f32(%v4f32) #0
declare %v4f32 @llvm.trunc.v4f32(%v4f32) #0
declare %v4f32 @llvm.rint.v4f32(%v4f32) #0
declare %v4f32 @llvm.nearbyint.v4f32(%v4f32) #0

;;; Double vector

%v2f64 = type <2 x double>
; CHECK: test_v2f64.sqrt:
define %v2f64 @test_v2f64.sqrt(%v2f64 %a) {
  ; CHECK: fsqrt.2d
  %1 = call %v2f64 @llvm.sqrt.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.powi:
define %v2f64 @test_v2f64.powi(%v2f64 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v2f64 @llvm.powi.v2f64(%v2f64 %a, i32 %b)
  ret %v2f64 %1
}
; CHECK: test_v2f64.sin:
define %v2f64 @test_v2f64.sin(%v2f64 %a) {
  ; CHECK: sin
  %1 = call %v2f64 @llvm.sin.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.cos:
define %v2f64 @test_v2f64.cos(%v2f64 %a) {
  ; CHECK: cos
  %1 = call %v2f64 @llvm.cos.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.pow:
define %v2f64 @test_v2f64.pow(%v2f64 %a, %v2f64 %b) {
  ; CHECK: pow
  %1 = call %v2f64 @llvm.pow.v2f64(%v2f64 %a, %v2f64 %b)
  ret %v2f64 %1
}
; CHECK: test_v2f64.exp:
define %v2f64 @test_v2f64.exp(%v2f64 %a) {
  ; CHECK: exp
  %1 = call %v2f64 @llvm.exp.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.exp2:
define %v2f64 @test_v2f64.exp2(%v2f64 %a) {
  ; CHECK: exp
  %1 = call %v2f64 @llvm.exp2.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.log:
define %v2f64 @test_v2f64.log(%v2f64 %a) {
  ; CHECK: log
  %1 = call %v2f64 @llvm.log.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.log10:
define %v2f64 @test_v2f64.log10(%v2f64 %a) {
  ; CHECK: log
  %1 = call %v2f64 @llvm.log10.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.log2:
define %v2f64 @test_v2f64.log2(%v2f64 %a) {
  ; CHECK: log
  %1 = call %v2f64 @llvm.log2.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.fma:
define %v2f64 @test_v2f64.fma(%v2f64 %a, %v2f64 %b, %v2f64 %c) {
  ; CHECK: fma
  %1 = call %v2f64 @llvm.fma.v2f64(%v2f64 %a, %v2f64 %b, %v2f64 %c)
  ret %v2f64 %1
}
; CHECK: test_v2f64.fabs:
define %v2f64 @test_v2f64.fabs(%v2f64 %a) {
  ; CHECK: fabs
  %1 = call %v2f64 @llvm.fabs.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.floor:
define %v2f64 @test_v2f64.floor(%v2f64 %a) {
  ; CHECK: frintm.2d
  %1 = call %v2f64 @llvm.floor.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.ceil:
define %v2f64 @test_v2f64.ceil(%v2f64 %a) {
  ; CHECK: frintp.2d
  %1 = call %v2f64 @llvm.ceil.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.trunc:
define %v2f64 @test_v2f64.trunc(%v2f64 %a) {
  ; CHECK: frintz.2d
  %1 = call %v2f64 @llvm.trunc.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.rint:
define %v2f64 @test_v2f64.rint(%v2f64 %a) {
  ; CHECK: frintx.2d
  %1 = call %v2f64 @llvm.rint.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.nearbyint:
define %v2f64 @test_v2f64.nearbyint(%v2f64 %a) {
  ; CHECK: frinti.2d
  %1 = call %v2f64 @llvm.nearbyint.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

declare %v2f64 @llvm.sqrt.v2f64(%v2f64) #0
declare %v2f64 @llvm.powi.v2f64(%v2f64, i32) #0
declare %v2f64 @llvm.sin.v2f64(%v2f64) #0
declare %v2f64 @llvm.cos.v2f64(%v2f64) #0
declare %v2f64 @llvm.pow.v2f64(%v2f64, %v2f64) #0
declare %v2f64 @llvm.exp.v2f64(%v2f64) #0
declare %v2f64 @llvm.exp2.v2f64(%v2f64) #0
declare %v2f64 @llvm.log.v2f64(%v2f64) #0
declare %v2f64 @llvm.log10.v2f64(%v2f64) #0
declare %v2f64 @llvm.log2.v2f64(%v2f64) #0
declare %v2f64 @llvm.fma.v2f64(%v2f64, %v2f64, %v2f64) #0
declare %v2f64 @llvm.fabs.v2f64(%v2f64) #0
declare %v2f64 @llvm.floor.v2f64(%v2f64) #0
declare %v2f64 @llvm.ceil.v2f64(%v2f64) #0
declare %v2f64 @llvm.trunc.v2f64(%v2f64) #0
declare %v2f64 @llvm.rint.v2f64(%v2f64) #0
declare %v2f64 @llvm.nearbyint.v2f64(%v2f64) #0

attributes #0 = { nounwind readonly }
