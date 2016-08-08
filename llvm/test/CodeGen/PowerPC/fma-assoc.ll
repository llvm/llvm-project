; RUN: llc -verify-machineinstrs < %s -march=ppc32 -fp-contract=fast -mattr=-vsx -disable-ppc-vsx-fma-mutation=false | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -fp-contract=fast -mattr=+vsx -mcpu=pwr7 -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK-VSX %s

define double @test_FMADD_ASSOC1(double %A, double %B, double %C,
                                 double %D, double %E) {
  %F = fmul double %A, %B         ; <double> [#uses=1]
  %G = fmul double %C, %D         ; <double> [#uses=1]
  %H = fadd double %F, %G         ; <double> [#uses=1]
  %I = fadd double %H, %E         ; <double> [#uses=1]
  ret double %I
; CHECK-LABEL: test_FMADD_ASSOC1:
; CHECK: fmadd
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD_ASSOC1:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD_ASSOC2(double %A, double %B, double %C,
                                 double %D, double %E) {
  %F = fmul double %A, %B         ; <double> [#uses=1]
  %G = fmul double %C, %D         ; <double> [#uses=1]
  %H = fadd double %F, %G         ; <double> [#uses=1]
  %I = fadd double %E, %H         ; <double> [#uses=1]
  ret double %I
; CHECK-LABEL: test_FMADD_ASSOC2:
; CHECK: fmadd
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD_ASSOC2:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC1(double %A, double %B, double %C,
                                 double %D, double %E) {
  %F = fmul double %A, %B         ; <double> [#uses=1]
  %G = fmul double %C, %D         ; <double> [#uses=1]
  %H = fadd double %F, %G         ; <double> [#uses=1]
  %I = fsub double %H, %E         ; <double> [#uses=1]
  ret double %I
; CHECK-LABEL: test_FMSUB_ASSOC1:
; CHECK: fmsub
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_ASSOC1:
; CHECK-VSX: xsmsubmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC2(double %A, double %B, double %C,
                                 double %D, double %E) {
  %F = fmul double %A, %B         ; <double> [#uses=1]
  %G = fmul double %C, %D         ; <double> [#uses=1]
  %H = fadd double %F, %G         ; <double> [#uses=1]
  %I = fsub double %E, %H         ; <double> [#uses=1]
  ret double %I
; CHECK-LABEL: test_FMSUB_ASSOC2:
; CHECK: fnmsub
; CHECK-NEXT: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_ASSOC2:
; CHECK-VSX: xsnmsubmdp
; CHECK-VSX-NEXT: xsnmsubadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD_ASSOC_EXT1(float %A, float %B, double %C,
                                 double %D, double %E) {
  %F = fmul float %A, %B         ; <float> [#uses=1]
  %G = fpext float %F to double   ; <double> [#uses=1]
  %H = fmul double %C, %D         ; <double> [#uses=1]
  %I = fadd double %H, %G         ; <double> [#uses=1]
  %J = fadd double %I, %E         ; <double> [#uses=1]
  ret double %J
; CHECK-LABEL: test_FMADD_ASSOC_EXT1:
; CHECK: fmadd
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD_ASSOC_EXT1:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD_ASSOC_EXT2(float %A, float %B, float %C,
                                 float %D, double %E) {
  %F = fmul float %A, %B         ; <float> [#uses=1]
  %G = fmul float %C, %D         ; <float> [#uses=1]
  %H = fadd float %F, %G         ; <float> [#uses=1]
  %I = fpext float %H to double   ; <double> [#uses=1]
  %J = fadd double %I, %E         ; <double> [#uses=1]
  ret double %J
; CHECK-LABEL: test_FMADD_ASSOC_EXT2:
; CHECK: fmadd
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD_ASSOC_EXT2:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD_ASSOC_EXT3(float %A, float %B, double %C,
                                 double %D, double %E) {
  %F = fmul float %A, %B          ; <float> [#uses=1]
  %G = fpext float %F to double   ; <double> [#uses=1]
  %H = fmul double %C, %D         ; <double> [#uses=1]
  %I = fadd double %H, %G         ; <double> [#uses=1]
  %J = fadd double %E, %I         ; <double> [#uses=1]
  ret double %J
; CHECK-LABEL: test_FMADD_ASSOC_EXT3:
; CHECK: fmadd
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD_ASSOC_EXT3:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD_ASSOC_EXT4(float %A, float %B, float %C,
                                 float %D, double %E) {
  %F = fmul float %A, %B          ; <float> [#uses=1]
  %G = fmul float %C, %D          ; <float> [#uses=1]
  %H = fadd float %F, %G          ; <float> [#uses=1]
  %I = fpext float %H to double   ; <double> [#uses=1]
  %J = fadd double %E, %I         ; <double> [#uses=1]
  ret double %J
; CHECK-LABEL: test_FMADD_ASSOC_EXT4:
; CHECK: fmadd
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD_ASSOC_EXT4:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC_EXT1(float %A, float %B, double %C,
                                 double %D, double %E) {
  %F = fmul float %A, %B          ; <float> [#uses=1]
  %G = fpext float %F to double   ; <double> [#uses=1]
  %H = fmul double %C, %D         ; <double> [#uses=1]
  %I = fadd double %H, %G         ; <double> [#uses=1]
  %J = fsub double %I, %E         ; <double> [#uses=1]
  ret double %J
; CHECK-LABEL: test_FMSUB_ASSOC_EXT1:
; CHECK: fmsub
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_ASSOC_EXT1:
; CHECK-VSX: xsmsubmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC_EXT2(float %A, float %B, float %C,
                                 float %D, double %E) {
  %F = fmul float %A, %B          ; <float> [#uses=1]
  %G = fmul float %C, %D          ; <float> [#uses=1]
  %H = fadd float %F, %G          ; <float> [#uses=1]
  %I = fpext float %H to double   ; <double> [#uses=1]
  %J = fsub double %I, %E         ; <double> [#uses=1]
  ret double %J
; CHECK-LABEL: test_FMSUB_ASSOC_EXT2:
; CHECK: fmsub
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_ASSOC_EXT2:
; CHECK-VSX: xsmsubmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC_EXT3(float %A, float %B, double %C,
                                 double %D, double %E) {
  %F = fmul float %A, %B          ; <float> [#uses=1]
  %G = fpext float %F to double   ; <double> [#uses=1]
  %H = fmul double %C, %D         ; <double> [#uses=1]
  %I = fadd double %H, %G         ; <double> [#uses=1]
  %J = fsub double %E, %I         ; <double> [#uses=1]
  ret double %J
; CHECK-LABEL: test_FMSUB_ASSOC_EXT3:
; CHECK: fnmsub
; CHECK-NEXT: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_ASSOC_EXT3:
; CHECK-VSX: xsnmsubmdp
; CHECK-VSX-NEXT: xsnmsubadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC_EXT4(float %A, float %B, float %C,
                                 float %D, double %E) {
  %F = fmul float %A, %B          ; <float> [#uses=1]
  %G = fmul float %C, %D          ; <float> [#uses=1]
  %H = fadd float %F, %G          ; <float> [#uses=1]
  %I = fpext float %H to double   ; <double> [#uses=1]
  %J = fsub double %E, %I         ; <double> [#uses=1]
  ret double %J
; CHECK-LABEL: test_FMSUB_ASSOC_EXT4:
; CHECK: fnmsub
; CHECK-NEXT: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_ASSOC_EXT4:
; CHECK-VSX: xsnmsubmdp
; CHECK-VSX-NEXT: xsnmsubadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}
