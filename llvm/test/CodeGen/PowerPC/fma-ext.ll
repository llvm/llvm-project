; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -fp-contract=fast -mattr=-vsx -disable-ppc-vsx-fma-mutation=false | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -fp-contract=fast -mattr=+vsx -mcpu=pwr7 -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK-VSX %s

define double @test_FMADD_EXT1(float %A, float %B, double %C) {
    %D = fmul float %A, %B          ; <float> [#uses=1]
    %E = fpext float %D to double   ; <double> [#uses=1]
    %F = fadd double %E, %C         ; <double> [#uses=1]
    ret double %F
; CHECK-LABEL: test_FMADD_EXT1:
; CHECK: fmadd
; CHECK-NEXT: blr
                                
; CHECK-VSX-LABEL: test_FMADD_EXT1:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD_EXT2(float %A, float %B, double %C) {
    %D = fmul float %A, %B          ; <float> [#uses=1]
    %E = fpext float %D to double   ; <double> [#uses=1]
    %F = fadd double %C, %E         ; <double> [#uses=1]
    ret double %F
; CHECK-LABEL: test_FMADD_EXT2:
; CHECK: fmadd
; CHECK-NEXT: blr
                                
; CHECK-VSX-LABEL: test_FMADD_EXT2:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_EXT1(float %A, float %B, double %C) {
    %D = fmul float %A, %B          ; <float> [#uses=1]
    %E = fpext float %D to double   ; <double> [#uses=1]
    %F = fsub double %E, %C         ; <double> [#uses=1]
    ret double %F
; CHECK-LABEL: test_FMSUB_EXT1:
; CHECK: fmsub
; CHECK-NEXT: blr
                                
; CHECK-VSX-LABEL: test_FMSUB_EXT1:
; CHECK-VSX: xsmsubmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_EXT2(float %A, float %B, double %C) {
    %D = fmul float %A, %B          ; <float> [#uses=1]
    %E = fpext float %D to double   ; <double> [#uses=1]
    %F = fsub double %C, %E         ; <double> [#uses=1]
    ret double %F
; CHECK-LABEL: test_FMSUB_EXT2:
; CHECK: fneg
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr
                                
; CHECK-VSX-LABEL: test_FMSUB_EXT2:
; CHECK-VSX: xsnegdp
; CHECK-VSX-NEXT: xsmaddmdp
; CHECK-VSX-NEXT: blr
}

; need nsz flag to generate fnmsub since it may affect sign of zero
define double @test_FMSUB_EXT2_NSZ(float %A, float %B, double %C) {
    %D = fmul nsz float %A, %B      ; <float> [#uses=1]
    %E = fpext float %D to double   ; <double> [#uses=1]
    %F = fsub nsz double %C, %E     ; <double> [#uses=1]
    ret double %F
; CHECK-LABEL: test_FMSUB_EXT2_NSZ:
; CHECK: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_EXT2_NSZ:
; CHECK-VSX: xsnmsubmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_EXT3(float %A, float %B, double %C) {
    %D = fmul float %A, %B          ; <float> [#uses=1]
    %E = fsub float -0.000000e+00, %D ;    <float> [#uses=1]
    %F = fpext float %E to double   ; <double> [#uses=1]
    %G = fsub double %F, %C         ; <double> [#uses=1]
    ret double %G
; CHECK-LABEL: test_FMSUB_EXT3:
; CHECK: fnmadd

; CHECK-NEXT: blr
                                
; CHECK-VSX-LABEL: test_FMSUB_EXT3:
; CHECK-VSX: xsnmaddmdp

; CHECK-VSX-NEXT: blr
}
    
define double @test_FMSUB_EXT4(float %A, float %B, double %C) {
    %D = fmul float %A, %B          ; <float> [#uses=1]
    %E = fpext float %D to double   ; <double> [#uses=1]
    %F = fsub double -0.000000e+00, %E ;    <double> [#uses=1]
    %G = fsub double %F, %C         ; <double> [#uses=1]
    ret double %G
; CHECK-LABEL: test_FMSUB_EXT4:
; CHECK: fnmadd

; CHECK-NEXT: blr
                                
; CHECK-VSX-LABEL: test_FMSUB_EXT4:
; CHECK-VSX: xsnmaddmdp

; CHECK-VSX-NEXT: blr
}  
