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

define %v4f16 @test_v4f16.powi(%v4f16 %a, i32 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v4f16.powi:
  ; CHECK-COUNT-4: bl __powi
  %1 = call %v4f16 @llvm.powi.v4f16.i32(%v4f16 %a, i32 %b)
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

declare %v4f16 @llvm.powi.v4f16.i32(%v4f16, i32) #0
declare %v4f16 @llvm.sin.v4f16(%v4f16) #0
declare %v4f16 @llvm.cos.v4f16(%v4f16) #0
declare %v4f16 @llvm.exp.v4f16(%v4f16) #0
declare %v4f16 @llvm.exp2.v4f16(%v4f16) #0
declare %v4f16 @llvm.log.v4f16(%v4f16) #0
declare %v4f16 @llvm.log10.v4f16(%v4f16) #0
declare %v4f16 @llvm.log2.v4f16(%v4f16) #0
declare %v4f16 @llvm.fma.v4f16(%v4f16, %v4f16, %v4f16) #0

;;;

%v8f16 = type <8 x half>

define %v8f16 @test_v8f16.powi(%v8f16 %a, i32 %b) {
  ; This operation is expanded, whether with or without +fullfp16.
  ; CHECK-LABEL:   test_v8f16.powi:
  ; CHECK-COUNT-8: bl __powi
  ; GISEL-LABEL:   test_v8f16.powi:
  ; GISEL-COUNT-8: bl __powi
  %1 = call %v8f16 @llvm.powi.v8f16.i32(%v8f16 %a, i32 %b)
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

declare %v8f16 @llvm.powi.v8f16.i32(%v8f16, i32) #0
declare %v8f16 @llvm.sin.v8f16(%v8f16) #0
declare %v8f16 @llvm.cos.v8f16(%v8f16) #0
declare %v8f16 @llvm.exp.v8f16(%v8f16) #0
declare %v8f16 @llvm.exp2.v8f16(%v8f16) #0
declare %v8f16 @llvm.log.v8f16(%v8f16) #0
declare %v8f16 @llvm.log10.v8f16(%v8f16) #0
declare %v8f16 @llvm.log2.v8f16(%v8f16) #0
declare %v8f16 @llvm.fma.v8f16(%v8f16, %v8f16, %v8f16) #0

;;; Float vectors

%v2f32 = type <2 x float>

; CHECK: test_v2f32.powi:
define %v2f32 @test_v2f32.powi(%v2f32 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v2f32 @llvm.powi.v2f32.i32(%v2f32 %a, i32 %b)
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

declare %v2f32 @llvm.powi.v2f32.i32(%v2f32, i32) #0
declare %v2f32 @llvm.sin.v2f32(%v2f32) #0
declare %v2f32 @llvm.cos.v2f32(%v2f32) #0
declare %v2f32 @llvm.exp.v2f32(%v2f32) #0
declare %v2f32 @llvm.exp2.v2f32(%v2f32) #0
declare %v2f32 @llvm.log.v2f32(%v2f32) #0
declare %v2f32 @llvm.log10.v2f32(%v2f32) #0
declare %v2f32 @llvm.log2.v2f32(%v2f32) #0
declare %v2f32 @llvm.fma.v2f32(%v2f32, %v2f32, %v2f32) #0

;;;

%v4f32 = type <4 x float>

; CHECK: test_v4f32.powi:
define %v4f32 @test_v4f32.powi(%v4f32 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v4f32 @llvm.powi.v4f32.i32(%v4f32 %a, i32 %b)
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

declare %v4f32 @llvm.powi.v4f32.i32(%v4f32, i32) #0
declare %v4f32 @llvm.sin.v4f32(%v4f32) #0
declare %v4f32 @llvm.cos.v4f32(%v4f32) #0
declare %v4f32 @llvm.exp.v4f32(%v4f32) #0
declare %v4f32 @llvm.exp2.v4f32(%v4f32) #0
declare %v4f32 @llvm.log.v4f32(%v4f32) #0
declare %v4f32 @llvm.log10.v4f32(%v4f32) #0
declare %v4f32 @llvm.log2.v4f32(%v4f32) #0
declare %v4f32 @llvm.fma.v4f32(%v4f32, %v4f32, %v4f32) #0

;;; Double vector

%v2f64 = type <2 x double>

; CHECK: test_v2f64.powi:
define %v2f64 @test_v2f64.powi(%v2f64 %a, i32 %b) {
  ; CHECK: pow
  %1 = call %v2f64 @llvm.powi.v2f64.i32(%v2f64 %a, i32 %b)
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

declare %v2f64 @llvm.powi.v2f64.i32(%v2f64, i32) #0
declare %v2f64 @llvm.sin.v2f64(%v2f64) #0
declare %v2f64 @llvm.cos.v2f64(%v2f64) #0
declare %v2f64 @llvm.exp.v2f64(%v2f64) #0
declare %v2f64 @llvm.exp2.v2f64(%v2f64) #0
declare %v2f64 @llvm.log.v2f64(%v2f64) #0
declare %v2f64 @llvm.log10.v2f64(%v2f64) #0
declare %v2f64 @llvm.log2.v2f64(%v2f64) #0
declare %v2f64 @llvm.fma.v2f64(%v2f64, %v2f64, %v2f64) #0

attributes #0 = { nounwind readonly }
