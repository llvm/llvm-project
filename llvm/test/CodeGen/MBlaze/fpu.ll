; Ensure that floating point operations are lowered to function calls when the
; FPU is not available in the hardware and that function calls are not used
; when the FPU is available in the hardware.
;
; RUN: llc < %s -march=mblaze | FileCheck -check-prefix=FUN %s
; RUN: llc < %s -march=mblaze -mattr=+fpu | FileCheck -check-prefix=FPU %s

define float @test_add(float %a, float %b) {
    ; FUN:        test_add:
    ; FPU:        test_add:

    %tmp.1 = fadd float %a, %b
    ; FUN:        brlid
    ; FPU-NOT:    brlid

    ret float %tmp.1
    ; FUN:        rtsd
    ; FPU:        rtsd
    ; FUN-NOT:    fadd
    ; FPU-NEXT:   fadd
}

define float @test_sub(float %a, float %b) {
    ; FUN:        test_sub:
    ; FPU:        test_sub:

    %tmp.1 = fsub float %a, %b
    ; FUN:        brlid
    ; FPU-NOT:    brlid

    ret float %tmp.1
    ; FUN:        rtsd
    ; FPU:        rtsd
    ; FUN-NOT:    frsub
    ; FPU-NEXT:   frsub
}

define float @test_mul(float %a, float %b) {
    ; FUN:        test_mul:
    ; FPU:        test_mul:

    %tmp.1 = fmul float %a, %b
    ; FUN:        brlid
    ; FPU-NOT:    brlid

    ret float %tmp.1
    ; FUN:        rtsd
    ; FPU:        rtsd
    ; FUN-NOT:    fmul
    ; FPU-NEXT:   fmul
}

define float @test_div(float %a, float %b) {
    ; FUN:        test_div:
    ; FPU:        test_div:

    %tmp.1 = fdiv float %a, %b
    ; FUN:        brlid
    ; FPU-NOT:    brlid

    ret float %tmp.1
    ; FUN:        rtsd
    ; FPU:        rtsd
    ; FUN-NOT:    fdiv
    ; FPU-NEXT:   fdiv
}
