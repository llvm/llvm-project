; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mattr=-fullfp16 \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOFP16
; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mattr=+fullfp16 \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FP16

; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mattr=-fullfp16 \
; RUN:     -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* \
; RUN:     2>&1 | FileCheck %s --check-prefixes=GISEL,GISEL-NOFP16,FALLBACK
; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mattr=+fullfp16 \
; RUN:     -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* \
; RUN:     2>&1 | FileCheck %s --check-prefixes=GISEL,GISEL-FP16,FALLBACK

;;; Half vectors

%v4f16 = type <4 x half>

; FALLBACK-NOT: remark{{.*}}test_v4f16.sqrt
define %v4f16 @test_v4f16.sqrt(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.sqrt:
  ; CHECK-NOFP16-COUNT-4: fsqrt s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fsqrt.4h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v4f16.sqrt:
  ; GISEL-NOFP16-COUNT-4: fsqrt s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           fsqrt.4h
  ; GISEL-FP16-NEXT:      ret
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

; FALLBACK-NOT: remark{{.*}}test_v4f16.sin
define %v4f16 @test_v4f16.sin(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.sin:
  ; CHECK-COUNT-4: bl sinf
  ; GISEL-LABEL:   test_v4f16.sin:
  ; GISEL-COUNT-4: bl sinf
  %1 = call %v4f16 @llvm.sin.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.cos
define %v4f16 @test_v4f16.cos(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.cos:
  ; CHECK-COUNT-4: bl cosf
  ; GISEL-LABEL:   test_v4f16.cos:
  ; GISEL-COUNT-4: bl cosf
  %1 = call %v4f16 @llvm.cos.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.pow
define %v4f16 @test_v4f16.pow(%v4f16 %a, %v4f16 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.pow:
  ; GISEL-LABEL:   test_v4f16.pow:
  ; CHECK-COUNT-4: bl pow
  ; GISEL-COUNT-4: bl pow
  %1 = call %v4f16 @llvm.pow.v4f16(%v4f16 %a, %v4f16 %b)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.exp
define %v4f16 @test_v4f16.exp(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.exp:
  ; CHECK-COUNT-4: bl exp
  ; GISEL-LABEL:   test_v4f16.exp:
  ; GISEL-COUNT-4: bl exp
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

; FALLBACK-NOT: remark{{.*}}test_v4f16.log
define %v4f16 @test_v4f16.log(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.log:
  ; CHECK-COUNT-4: bl log
  ; GISEL-LABEL:   test_v4f16.log:
  ; GISEL-COUNT-4: bl log
  %1 = call %v4f16 @llvm.log.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.log10
define %v4f16 @test_v4f16.log10(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.log10:
  ; CHECK-COUNT-4: bl log10
  ; GISEL-LABEL:   test_v4f16.log10:
  ; GISEL-COUNT-4: bl log10
  %1 = call %v4f16 @llvm.log10.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.log2
define %v4f16 @test_v4f16.log2(%v4f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.log2:
  ; CHECK-COUNT-4: bl log2
  ; GISEL-LABEL:   test_v4f16.log2:
  ; GISEL-COUNT-4: bl log2
  %1 = call %v4f16 @llvm.log2.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.fma
define %v4f16 @test_v4f16.fma(%v4f16 %a, %v4f16 %b, %v4f16 %c) {
  ; CHECK-LABEL:          test_v4f16.fma:
  ; CHECK-NOFP16-COUNT-4: fmadd s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fmla.4h
  ; GISEL-LABEL:          test_v4f16.fma:
  ; GISEL-NOFP16-COUNT-4: fmadd s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           fmla.4h
  %1 = call %v4f16 @llvm.fma.v4f16(%v4f16 %a, %v4f16 %b, %v4f16 %c)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.fabs
define %v4f16 @test_v4f16.fabs(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.fabs:
  ; CHECK-NOFP16-COUNT-4: fabs s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fabs.4h
  ; CHECK-FP16-NEXT:      ret

  ; GISEL-LABEL:          test_v4f16.fabs:
  ; GISEL-NOFP16-COUNT-4: fabs s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           fabs.4h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.fabs.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.floor
define %v4f16 @test_v4f16.floor(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.floor:
  ; CHECK-NOFP16-COUNT-4: frintm s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintm.4h
  ; CHECK-FP16-NEXT:      ret

  ; GISEL-LABEL:          test_v4f16.floor:
  ; GISEL-NOFP16-COUNT-4: frintm s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frintm.4h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.floor.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.ceil(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.ceil:
  ; CHECK-NOFP16-COUNT-4: frintp s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintp.4h
  ; CHECK-FP16-NEXT:      ret
  ; FALLBACK-NOT: remark{{.*}}test_v4f16.ceil:
  ; GISEL-LABEL:          test_v4f16.ceil:
  ; GISEL-NOFP16-COUNT-4: frintp s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frintp.4h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.ceil.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.trunc
define %v4f16 @test_v4f16.trunc(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.trunc:
  ; CHECK-NOFP16-COUNT-4: frintz s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintz.4h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v4f16.trunc:
  ; GISEL-NOFP16-COUNT-4: frintz s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frintz.4h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.trunc.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.rint
define %v4f16 @test_v4f16.rint(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.rint:
  ; CHECK-NOFP16-COUNT-4: frintx s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintx.4h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v4f16.rint:
  ; GISEL-NOFP16-COUNT-4: frintx s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frintx.4h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.rint.v4f16(%v4f16 %a)
  ret %v4f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f16.nearbyint
define %v4f16 @test_v4f16.nearbyint(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.nearbyint:
  ; CHECK-NOFP16-COUNT-4: frinti s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frinti.4h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v4f16.nearbyint:
  ; GISEL-NOFP16-COUNT-4: frinti s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frinti.4h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v4f16 @llvm.nearbyint.v4f16(%v4f16 %a)
  ret %v4f16 %1
}
define %v4f16 @test_v4f16.round(%v4f16 %a) {
  ; CHECK-LABEL:          test_v4f16.round:
  ; CHECK-NOFP16-COUNT-4: frinta s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frinta.4h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v4f16.round:
  ; GISEL-NOFP16-COUNT-4: frinta s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frinta.4h
  ; GISEL-FP16-NEXT:      ret
  %1 =  call %v4f16 @llvm.round.v4f16(%v4f16 %a)
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
declare %v4f16 @llvm.round.v4f16(%v4f16) #0

;;;

%v8f16 = type <8 x half>

; FALLBACK-NOT: remark{{.*}}test_v8f16.sqrt
define %v8f16 @test_v8f16.sqrt(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.sqrt:
  ; CHECK-NOFP16-COUNT-8: fsqrt s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fsqrt.8h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v8f16.sqrt:
  ; GISEL-NOFP16-COUNT-8: fsqrt s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           fsqrt.8h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.sqrt.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.powi(%v8f16 %a, i32 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.powi:
  ; CHECK-COUNT-8: bl __powi
  ; GISEL-LABEL:   test_v8f16.powi:
  ; GISEL-COUNT-8: bl __powi
  %1 = call %v8f16 @llvm.powi.v8f16(%v8f16 %a, i32 %b)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.sin
define %v8f16 @test_v8f16.sin(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.sin:
  ; CHECK-COUNT-8: bl sinf
  ; GISEL-LABEL:   test_v8f16.sin:
  ; GISEL-COUNT-8: bl sinf
  %1 = call %v8f16 @llvm.sin.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.cos
define %v8f16 @test_v8f16.cos(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.cos:
  ; CHECK-COUNT-8: bl cosf
  ; GISEL-LABEL:   test_v8f16.cos:
  ; GISEL-COUNT-8: bl cosf
  %1 = call %v8f16 @llvm.cos.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.pow
define %v8f16 @test_v8f16.pow(%v8f16 %a, %v8f16 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.pow:
  ; CHECK-COUNT-8: bl pow
  ; GISEL-LABEL:   test_v8f16.pow:
  ; GISEL-COUNT-8: bl pow
  %1 = call %v8f16 @llvm.pow.v8f16(%v8f16 %a, %v8f16 %b)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.exp
define %v8f16 @test_v8f16.exp(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.exp:
  ; CHECK-COUNT-8: bl exp
  ; GISEL-LABEL:   test_v8f16.exp:
  ; GISEL-COUNT-8: bl exp
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

; FALLBACK-NOT: remark{{.*}}test_v8f16.log
define %v8f16 @test_v8f16.log(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.log:
  ; CHECK-COUNT-8: bl log
  ; GISEL-LABEL:   test_v8f16.log:
  ; GISEL-COUNT-8: bl log
  %1 = call %v8f16 @llvm.log.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.log10
define %v8f16 @test_v8f16.log10(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.log10:
  ; CHECK-COUNT-8: bl log10
  ; GISEL-LABEL:   test_v8f16.log10:
  ; GISEL-COUNT-8: bl log10
  %1 = call %v8f16 @llvm.log10.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.log2
define %v8f16 @test_v8f16.log2(%v8f16 %a) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.log2:
  ; CHECK-COUNT-8: bl log2
  ; GISEL-LABEL:   test_v8f16.log2:
  ; GISEL-COUNT-8: bl log2
  %1 = call %v8f16 @llvm.log2.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.fma
define %v8f16 @test_v8f16.fma(%v8f16 %a, %v8f16 %b, %v8f16 %c) {
  ; CHECK-LABEL:          test_v8f16.fma:
  ; CHECK-NOFP16-COUNT-8: fmadd s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fmla.8h
  ; GISEL-LABEL:          test_v8f16.fma:
  ; GISEL-NOFP16-COUNT-8: fmadd s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           fmla.8h
  %1 = call %v8f16 @llvm.fma.v8f16(%v8f16 %a, %v8f16 %b, %v8f16 %c)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.fabs
define %v8f16 @test_v8f16.fabs(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.fabs:
  ; CHECK-NOFP16-COUNT-8: fabs s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           fabs.8h
  ; CHECK-FP16-NEXT:      ret

  ; GISEL-LABEL:          test_v8f16.fabs:
  ; GISEL-NOFP16-COUNT-8: fabs s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           fabs.8h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.fabs.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.floor
define %v8f16 @test_v8f16.floor(%v8f16 %a) {
  ; CHECK-LABEL:     		  test_v8f16.floor:
  ; CHECK-NOFP16-COUNT-8: frintm s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintm.8h
  ; CHECK-FP16-NEXT:      ret

  ; GISEL-LABEL:     		  test_v8f16.floor:
  ; GISEL-NOFP16-COUNT-8: frintm s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frintm.8h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.floor.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.ceil(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.ceil:
  ; CHECK-NOFP16-COUNT-8: frintp s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintp.8h
  ; CHECK-FP16-NEXT:      ret
  ; FALLBACK-NOT:         remark{{.*}}test_v8f16.ceil:
  ; GISEL-LABEL:          test_v8f16.ceil:
  ; GISEL-NOFP16-COUNT-8: frintp s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frintp.8h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.ceil.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.trunc
define %v8f16 @test_v8f16.trunc(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.trunc:
  ; CHECK-NOFP16-COUNT-8: frintz s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintz.8h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v8f16.trunc:
  ; GISEL-NOFP16-COUNT-8: frintz s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frintz.8h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.trunc.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.rint
define %v8f16 @test_v8f16.rint(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.rint:
  ; CHECK-NOFP16-COUNT-8: frintx s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frintx.8h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v8f16.rint:
  ; GISEL-NOFP16-COUNT-8: frintx s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frintx.8h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.rint.v8f16(%v8f16 %a)
  ret %v8f16 %1
}

; FALLBACK-NOT: remark{{.*}}test_v8f16.nearbyint
define %v8f16 @test_v8f16.nearbyint(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.nearbyint:
  ; CHECK-NOFP16-COUNT-8: frinti s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frinti.8h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v8f16.nearbyint:
  ; GISEL-NOFP16-COUNT-8: frinti s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frinti.8h
  ; GISEL-FP16-NEXT:      ret
  %1 = call %v8f16 @llvm.nearbyint.v8f16(%v8f16 %a)
  ret %v8f16 %1
}
define %v8f16 @test_v8f16.round(%v8f16 %a) {
  ; CHECK-LABEL:          test_v8f16.round:
  ; CHECK-NOFP16-COUNT-8: frinta s{{[0-9]+}}, s{{[0-9]+}}
  ; CHECK-FP16-NOT:       fcvt
  ; CHECK-FP16:           frinta.8h
  ; CHECK-FP16-NEXT:      ret
  ; GISEL-LABEL:          test_v8f16.round:
  ; GISEL-NOFP16-COUNT-8: frinta s{{[0-9]+}}, s{{[0-9]+}}
  ; GISEL-FP16-NOT:       fcvt
  ; GISEL-FP16:           frinta.8h
  ; GISEL-FP16-NEXT:      ret
  %1 =  call %v8f16 @llvm.round.v8f16(%v8f16 %a)
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
declare %v8f16 @llvm.round.v8f16(%v8f16) #0

;;; Float vectors

%v2f32 = type <2 x float>

; FALLBACK-NOT: remark{{.*}}test_v2f32.sqrt
; CHECK-LABEL: test_v2f32.sqrt:
; GISEL-LABEL: test_v2f32.sqrt:
define %v2f32 @test_v2f32.sqrt(%v2f32 %a) {
  ; CHECK: fsqrt.2s
  ; GISEL: fsqrt.2s
  %1 = call %v2f32 @llvm.sqrt.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.powi:
define %v2f32 @test_v2f32.powi(%v2f32 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v2f32 @llvm.powi.v2f32(%v2f32 %a, i32 %b)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.sin
; CHECK: test_v2f32.sin:
define %v2f32 @test_v2f32.sin(%v2f32 %a) {
  ; CHECK: sin
  ; GISEL: sin
  %1 = call %v2f32 @llvm.sin.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.cos
; CHECK: test_v2f32.cos:
define %v2f32 @test_v2f32.cos(%v2f32 %a) {
  ; CHECK: cos
  ; GISEL: cos
  %1 = call %v2f32 @llvm.cos.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.pow
; CHECK: test_v2f32.pow:
; GISEL-LABEL: test_v2f32.pow:
define %v2f32 @test_v2f32.pow(%v2f32 %a, %v2f32 %b) {
  ; CHECK: pow
  ; GISEL: pow
  %1 = call %v2f32 @llvm.pow.v2f32(%v2f32 %a, %v2f32 %b)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.exp
; CHECK: test_v2f32.exp:
; GISEL: test_v2f32.exp:
define %v2f32 @test_v2f32.exp(%v2f32 %a) {
  ; CHECK: exp
  ; GISEL: exp
  %1 = call %v2f32 @llvm.exp.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK: test_v2f32.exp2:
define %v2f32 @test_v2f32.exp2(%v2f32 %a) {
  ; CHECK: exp
  %1 = call %v2f32 @llvm.exp2.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.log
; CHECK: test_v2f32.log:
define %v2f32 @test_v2f32.log(%v2f32 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v2f32 @llvm.log.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.log10
; CHECK: test_v2f32.log10:
; GISEL: test_v2f32.log10:
define %v2f32 @test_v2f32.log10(%v2f32 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v2f32 @llvm.log10.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.log2
; CHECK: test_v2f32.log2:
; GISEL: test_v2f32.log2:
define %v2f32 @test_v2f32.log2(%v2f32 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v2f32 @llvm.log2.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.fma
; CHECK-LABEL: test_v2f32.fma:
; GISEL-LABEL: test_v2f32.fma:
define %v2f32 @test_v2f32.fma(%v2f32 %a, %v2f32 %b, %v2f32 %c) {
  ; CHECK: fmla.2s
  ; GISEL: fmla.2s
  %1 = call %v2f32 @llvm.fma.v2f32(%v2f32 %a, %v2f32 %b, %v2f32 %c)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.fabs
; CHECK-LABEL: test_v2f32.fabs:
; GISEL-LABEL: test_v2f32.fabs:
define %v2f32 @test_v2f32.fabs(%v2f32 %a) {
  ; CHECK: fabs.2s
  ; GISEL: fabs.2s
  %1 = call %v2f32 @llvm.fabs.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.floor
; CHECK-LABEL: test_v2f32.floor:
; GISEL-LABEL: test_v2f32.floor:
define %v2f32 @test_v2f32.floor(%v2f32 %a) {
  ; CHECK: frintm.2s
  ; GISEL: frintm.2s
  %1 = call %v2f32 @llvm.floor.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.ceil:
; FALLBACK-NOT: remark{{.*}}test_v2f32.ceil
; GISEL-LABEL: test_v2f32.ceil:
define %v2f32 @test_v2f32.ceil(%v2f32 %a) {
  ; CHECK: frintp.2s
  ; GISEL: frintp.2s
  %1 = call %v2f32 @llvm.ceil.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.trunc:
; FALLBACK-NOT: remark{{.*}}test_v2f32.trunc
; GISEL-LABEL: test_v2f32.trunc:
define %v2f32 @test_v2f32.trunc(%v2f32 %a) {
  ; CHECK: frintz.2s
  ; GISEL: frintz.2s
  %1 = call %v2f32 @llvm.trunc.v2f32(%v2f32 %a)
  ret %v2f32 %1
}
; CHECK-LABEL: test_v2f32.rint:
; FALLBACK-NOT: remark{{.*}}test_v2f32.rint
; GISEL-LABEL: test_v2f32.rint:
define %v2f32 @test_v2f32.rint(%v2f32 %a) {
  ; CHECK: frintx.2s
  ; GISEL: frintx.2s
  %1 = call %v2f32 @llvm.rint.v2f32(%v2f32 %a)
  ret %v2f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f32.nearbyint
; CHECK-LABEL: test_v2f32.nearbyint:
; GISEL-LABEL: test_v2f32.nearbyint:
define %v2f32 @test_v2f32.nearbyint(%v2f32 %a) {
  ; CHECK: frinti.2s
  ; GISEL: frinti.2s
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

; FALLBACK-NOT: remark{{.*}}test_v4f32.sqrt
; CHECK: test_v4f32.sqrt:
; GISEL: test_v4f32.sqrt:
define %v4f32 @test_v4f32.sqrt(%v4f32 %a) {
  ; CHECK: fsqrt.4s
  ; GISEL: fsqrt.4s
  %1 = call %v4f32 @llvm.sqrt.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.powi:
define %v4f32 @test_v4f32.powi(%v4f32 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v4f32 @llvm.powi.v4f32(%v4f32 %a, i32 %b)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.sin
; CHECK: test_v4f32.sin:
define %v4f32 @test_v4f32.sin(%v4f32 %a) {
  ; CHECK: sin
  ; GISEL: sin
  %1 = call %v4f32 @llvm.sin.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.cos
; CHECK: test_v4f32.cos:
define %v4f32 @test_v4f32.cos(%v4f32 %a) {
  ; CHECK: cos
  ; GISEL: cos
  %1 = call %v4f32 @llvm.cos.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.pow
; CHECK: test_v4f32.pow:
; GISEL-LABEL: test_v4f32.pow:
define %v4f32 @test_v4f32.pow(%v4f32 %a, %v4f32 %b) {
  ; CHECK: pow
  ; GISEL: pow
  %1 = call %v4f32 @llvm.pow.v4f32(%v4f32 %a, %v4f32 %b)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.exp
; CHECK: test_v4f32.exp:
; GISEL: test_v4f32.exp:
define %v4f32 @test_v4f32.exp(%v4f32 %a) {
  ; CHECK: exp
  ; GISEL: exp
  %1 = call %v4f32 @llvm.exp.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.exp2:
define %v4f32 @test_v4f32.exp2(%v4f32 %a) {
  ; CHECK: exp
  %1 = call %v4f32 @llvm.exp2.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.log
; CHECK: test_v4f32.log:
define %v4f32 @test_v4f32.log(%v4f32 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v4f32 @llvm.log.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.log10
; CHECK: test_v4f32.log10:
define %v4f32 @test_v4f32.log10(%v4f32 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v4f32 @llvm.log10.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.log2
; CHECK: test_v4f32.log2:
; GISEL: test_v4f32.log2:
define %v4f32 @test_v4f32.log2(%v4f32 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v4f32 @llvm.log2.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.fma
; CHECK: test_v4f32.fma:
; GISEL: test_v4f32.fma:
define %v4f32 @test_v4f32.fma(%v4f32 %a, %v4f32 %b, %v4f32 %c) {
  ; CHECK: fma
  ; GISEL: fma
  %1 = call %v4f32 @llvm.fma.v4f32(%v4f32 %a, %v4f32 %b, %v4f32 %c)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.fabs
; CHECK: test_v4f32.fabs:
; GISEL: test_v4f32.fabs:
define %v4f32 @test_v4f32.fabs(%v4f32 %a) {
  ; CHECK: fabs
  ; GISEL: fabs
  %1 = call %v4f32 @llvm.fabs.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.floor
; CHECK: test_v4f32.floor:
; GISEL: test_v4f32.floor:
define %v4f32 @test_v4f32.floor(%v4f32 %a) {
  ; CHECK: frintm.4s
  ; GISEL: frintm.4s
  %1 = call %v4f32 @llvm.floor.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.ceil:
; FALLBACK-NOT: remark{{.*}}test_v4f32.ceil
; GISEL-LABEL: test_v4f32.ceil:
define %v4f32 @test_v4f32.ceil(%v4f32 %a) {
  ; CHECK: frintp.4s
  ; GISEL: frintp.4s
  %1 = call %v4f32 @llvm.ceil.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.trunc:
; FALLBACK-NOT: remark{{.*}}test_v4f32.trunc
; GISEL: test_v4f32.trunc:
define %v4f32 @test_v4f32.trunc(%v4f32 %a) {
  ; CHECK: frintz.4s
  ; GISEL: frintz.4s
  %1 = call %v4f32 @llvm.trunc.v4f32(%v4f32 %a)
  ret %v4f32 %1
}
; CHECK: test_v4f32.rint:
; FALLBACK-NOT: remark{{.*}}test_v4f32.rint
; GISEL: test_v4f32.rint:
define %v4f32 @test_v4f32.rint(%v4f32 %a) {
  ; CHECK: frintx.4s
  ; GISEL: frintx.4s
  %1 = call %v4f32 @llvm.rint.v4f32(%v4f32 %a)
  ret %v4f32 %1
}

; FALLBACK-NOT: remark{{.*}}test_v4f32.nearbyint
; CHECK: test_v4f32.nearbyint:
; GISEL: test_v4f32.nearbyint:
define %v4f32 @test_v4f32.nearbyint(%v4f32 %a) {
  ; CHECK: frinti.4s
  ; GISEL: frinti.4s
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
; FALLBACK-NOT: remark{{.*}}test_v2f64.sqrt
; CHECK: test_v2f64.sqrt:
; GISEL: test_v2f64.sqrt:
define %v2f64 @test_v2f64.sqrt(%v2f64 %a) {
  ; CHECK: fsqrt.2d
  ; GISEL: fsqrt.2d
  %1 = call %v2f64 @llvm.sqrt.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.powi:
define %v2f64 @test_v2f64.powi(%v2f64 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v2f64 @llvm.powi.v2f64(%v2f64 %a, i32 %b)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.sin
; CHECK: test_v2f64.sin:
define %v2f64 @test_v2f64.sin(%v2f64 %a) {
  ; CHECK: sin
  ; GISEL: sin
  %1 = call %v2f64 @llvm.sin.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.cos
; CHECK: test_v2f64.cos:
define %v2f64 @test_v2f64.cos(%v2f64 %a) {
  ; CHECK: cos
  ; GISEL: cos
  %1 = call %v2f64 @llvm.cos.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.pow
; CHECK: test_v2f64.pow:
; GISEL-LABEL: test_v2f64.pow:
define %v2f64 @test_v2f64.pow(%v2f64 %a, %v2f64 %b) {
  ; CHECK: pow
  ; GISEL: pow
  %1 = call %v2f64 @llvm.pow.v2f64(%v2f64 %a, %v2f64 %b)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.exp
; CHECK: test_v2f64.exp:
; GISEL: test_v2f64.exp:
define %v2f64 @test_v2f64.exp(%v2f64 %a) {
  ; CHECK: exp
  ; GISEL: exp
  %1 = call %v2f64 @llvm.exp.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.exp2:
define %v2f64 @test_v2f64.exp2(%v2f64 %a) {
  ; CHECK: exp
  %1 = call %v2f64 @llvm.exp2.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.log
; CHECK: test_v2f64.log:
define %v2f64 @test_v2f64.log(%v2f64 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v2f64 @llvm.log.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.log10
; CHECK: test_v2f64.log10:
; GISEL: test_v2f64.log10:
define %v2f64 @test_v2f64.log10(%v2f64 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v2f64 @llvm.log10.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.log2
; CHECK: test_v2f64.log2:
; GISEL: test_v2f64.log2:
define %v2f64 @test_v2f64.log2(%v2f64 %a) {
  ; CHECK: log
  ; GISEL: log
  %1 = call %v2f64 @llvm.log2.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.fma
; CHECK: test_v2f64.fma:
; GISEL: test_v2f64.fma:
define %v2f64 @test_v2f64.fma(%v2f64 %a, %v2f64 %b, %v2f64 %c) {
  ; CHECK: fma
  ; GISEL: fma
  %1 = call %v2f64 @llvm.fma.v2f64(%v2f64 %a, %v2f64 %b, %v2f64 %c)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.fabs
; CHECK: test_v2f64.fabs:
; GISEL: test_v2f64.fabs:
define %v2f64 @test_v2f64.fabs(%v2f64 %a) {
  ; CHECK: fabs
  ; GISEL: fabs
  %1 = call %v2f64 @llvm.fabs.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.floor
; CHECK: test_v2f64.floor:
; GISEL: test_v2f64.floor:
define %v2f64 @test_v2f64.floor(%v2f64 %a) {
  ; CHECK: frintm.2d
  ; GISEL: frintm.2d
  %1 = call %v2f64 @llvm.floor.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.ceil:
; FALLBACK-NOT: remark{{.*}}test_v2f64.ceil
; GISEL-LABEL: test_v2f64.ceil:
define %v2f64 @test_v2f64.ceil(%v2f64 %a) {
  ; CHECK: frintp.2d
  ; GISEL: frintp.2d
  %1 = call %v2f64 @llvm.ceil.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.trunc:
; FALLBACK-NOT: remark{{.*}}test_v2f64.trunc
; GISEL: test_v2f64.trunc:
define %v2f64 @test_v2f64.trunc(%v2f64 %a) {
  ; CHECK: frintz.2d
  ; GISEL: frintz.2d
  %1 = call %v2f64 @llvm.trunc.v2f64(%v2f64 %a)
  ret %v2f64 %1
}
; CHECK: test_v2f64.rint:
; FALLBACK-NOT: remark{{.*}}test_v2f64.rint
; GISEL: test_v2f64.rint:
define %v2f64 @test_v2f64.rint(%v2f64 %a) {
  ; CHECK: frintx.2d
  ; GISEL: frintx.2d
  %1 = call %v2f64 @llvm.rint.v2f64(%v2f64 %a)
  ret %v2f64 %1
}

; FALLBACK-NOT: remark{{.*}}test_v2f64.nearbyint
; CHECK: test_v2f64.nearbyint:
; GISEL: test_v2f64.nearbyint:
define %v2f64 @test_v2f64.nearbyint(%v2f64 %a) {
  ; CHECK: frinti.2d
  ; GISEL: frinti.2d
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
