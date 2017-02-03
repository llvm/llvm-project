; RUN: llc -verify-machineinstrs < %s -march=ppc32 -fp-contract=fast -mattr=-vsx -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SAFE %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -fp-contract=fast -mattr=+vsx -mcpu=pwr7 -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK-VSX -check-prefix=CHECK-VSX-SAFE %s
; RUN: llc -verify-machineinstrs < %s -march=ppc32 -fp-contract=fast -enable-unsafe-fp-math -mattr=-vsx -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK -check-prefix=CHECK-UNSAFE %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -fp-contract=fast -enable-unsafe-fp-math -mattr=+vsx -mcpu=pwr7 -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK-VSX -check-prefix=CHECK-UNSAFE-VSX %s

define double @test_FMADD_ASSOC1(double %A, double %B, double %C,
                                 double %D, double %E) {
  %F = fmul double %A, %B         ; <double> [#uses=1]
  %G = fmul double %C, %D         ; <double> [#uses=1]
  %H = fadd double %F, %G         ; <double> [#uses=1]
  %I = fadd double %H, %E         ; <double> [#uses=1]
  ret double %I
; CHECK-SAFE-LABEL: test_FMADD_ASSOC1:
; CHECK-SAFE: fmul
; CHECK-SAFE-NEXT: fmadd
; CHECK-SAFE-NEXT: fadd
; CHECK-SAFE-NEXT: blr

; CHECK-UNSAFE-LABEL: test_FMADD_ASSOC1:
; CHECK-UNSAFE: fmadd
; CHECK-UNSAFE-NEXT: fmadd
; CHECK-UNSAFE-NEXT: blr

; CHECK-VSX-SAFE-LABEL: test_FMADD_ASSOC1:
; CHECK-VSX-SAFE: xsmuldp
; CHECK-VSX-SAFE-NEXT: xsmaddadp
; CHECK-VSX-SAFE-NEXT: xsadddp
; CHECK-VSX-SAFE-NEXT: blr

; CHECK-VSX-UNSAFE-LABEL: test_FMADD_ASSOC1:
; CHECK-VSX-UNSAFE: xsmaddmdp
; CHECK-VSX-UNSAFE-NEXT: xsmaddadp
; CHECK-VSX-UNSAFE-NEXT: fmr
; CHECK-VSX-UNSAFE-NEXT: blr
}

define double @test_FMADD_ASSOC2(double %A, double %B, double %C,
                                 double %D, double %E) {
  %F = fmul double %A, %B         ; <double> [#uses=1]
  %G = fmul double %C, %D         ; <double> [#uses=1]
  %H = fadd double %F, %G         ; <double> [#uses=1]
  %I = fadd double %E, %H         ; <double> [#uses=1]
  ret double %I
; CHECK-SAFE-LABEL: test_FMADD_ASSOC2:
; CHECK-SAFE: fmul
; CHECK-SAFE-NEXT: fmadd
; CHECK-SAFE-NEXT: fadd
; CHECK-SAFE-NEXT: blr

; CHECK-UNSAFE-LABEL: test_FMADD_ASSOC2:
; CHECK-UNSAFE: fmadd
; CHECK-UNSAFE-NEXT: fmadd
; CHECK-UNSAFE-NEXT: blr

; CHECK-VSX-SAFE-LABEL: test_FMADD_ASSOC2:
; CHECK-VSX-SAFE: xsmuldp
; CHECK-VSX-SAFE-NEXT: xsmaddadp
; CHECK-VSX-SAFE-NEXT: xsadddp
; CHECK-VSX-SAFE-NEXT: blr

; CHECK-VSX-UNSAFE-LABEL: test_FMADD_ASSOC2:
; CHECK-VSX-UNSAFE: xsmaddmdp
; CHECK-VSX-UNSAFE-NEXT: xsmaddadp
; CHECK-VSX-UNSAFE-NEXT: fmr
; CHECK-VSX-UNSAFE-NEXT: blr
}

define double @test_FMSUB_ASSOC1(double %A, double %B, double %C,
                                 double %D, double %E) {
  %F = fmul double %A, %B         ; <double> [#uses=1]
  %G = fmul double %C, %D         ; <double> [#uses=1]
  %H = fadd double %F, %G         ; <double> [#uses=1]
  %I = fsub double %H, %E         ; <double> [#uses=1]
  ret double %I
; CHECK-SAFE-LABEL: test_FMSUB_ASSOC1:
; CHECK-SAFE: fmul
; CHECK-SAFE-NEXT: fmadd
; CHECK-SAFE-NEXT: fsub
; CHECK-SAFE-NEXT: blr

; CHECK-UNSAFE-LABEL: test_FMSUB_ASSOC1:
; CHECK-UNSAFE: fmsub
; CHECK-UNSAFE-NEXT: fmadd
; CHECK-UNSAFE-NEXT: blr

; CHECK-SAFE-VSX-LABEL: test_FMSUB_ASSOC1:
; CHECK-SAFE-VSX: xsmuldp
; CHECK-SAFE-VSX-NEXT: xsmaddadp
; CHECK-SAFE-VSX-NEXT: xssubdp
; CHECK-SAFE-VSX-NEXT: blr

; CHECK-UNSAFE-VSX-LABEL: test_FMSUB_ASSOC1:
; CHECK-UNSAFE-VSX: xsmsubmdp
; CHECK-UNSAFE-VSX-NEXT: xsmaddadp
; CHECK-UNSAFE-VSX-NEXT: fmr
; CHECK-UNSAFE-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC2(double %A, double %B, double %C,
                                 double %D, double %E) {
  %F = fmul double %A, %B         ; <double> [#uses=1]
  %G = fmul double %C, %D         ; <double> [#uses=1]
  %H = fadd double %F, %G         ; <double> [#uses=1]
  %I = fsub double %E, %H         ; <double> [#uses=1]
  ret double %I
; CHECK-SAFE-LABEL: test_FMSUB_ASSOC2:
; CHECK-SAFE: fmul
; CHECK-SAFE-NEXT: fmadd
; CHECK-SAFE-NEXT: fsub
; CHECK-SAFE-NEXT: blr

; CHECK-UNSAFE-LABEL: test_FMSUB_ASSOC2:
; CHECK-UNSAFE: fnmsub
; CHECK-UNSAFE-NEXT: fnmsub
; CHECK-UNSAFE-NEXT: blr

; CHECK-SAFE-VSX-LABEL: test_FMSUB_ASSOC2:
; CHECK-SAFE-VSX: xsmuldp
; CHECK-SAFE-VSX-NEXT: xsmaddadp
; CHECK-SAFE-VSX-NEXT: xssubdp
; CHECK-SAFE-VSX-NEXT: blr

; CHECK-UNSAFE-VSX-LABEL: test_FMSUB_ASSOC2:
; CHECK-UNSAFE-VSX: xsnmsubmdp
; CHECK-UNSAFE-VSX-NEXT: xsnmsubadp
; CHECK-UNSAFE-VSX-NEXT: fmr
; CHECK-UNSAFE-VSX-NEXT: blr
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
