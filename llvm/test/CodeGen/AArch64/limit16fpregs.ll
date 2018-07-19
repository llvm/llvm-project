; RUN: llc -o - %s | FileCheck %s
target triple="aarch64--"

; Make sure we are not using any of the s16-s31 fp registers.
; CHECK-LABEL: limit_16_fpregs:
; CHECK-NOT: {{[bhsdq]1[6789]}}
; CHECK-NOT: {{[bhsdq]2[0123456789]}}
; CHECK-NOT: {{[bhsdq]3[01]}}
define float @limit_16_fpregs(
  float %p00,
  float %p01,
  float %p02,
  float %p03,
  float %p04,
  float %p05,
  float %p06,
  float %p07,
  float %p08,
  float %p09,
  float %p10,
  float %p11,
  float %p12,
  float %p13,
  float %p14,
  float %p15,
  float %p16,
  float %p17,
  float %p18,
  float %p19,
  float %p20,
  float %p21,
  float %p22,
  float %p23,
  float %p24,
  float %p25,
  float %p26,
  float %p27,
  float %p28,
  float %p29,
  float %p30,
  float %p31
) #0 {
  %v00 = fadd float %p00, %p01
  %v01 = fadd float %p02, %p03
  %v02 = fadd float %p04, %p05
  %v03 = fadd float %p06, %p07
  %v04 = fadd float %p08, %p09
  %v05 = fadd float %p10, %p11
  %v06 = fadd float %p12, %p13
  %v07 = fadd float %p14, %p15
  %v08 = fadd float %p16, %p17
  %v09 = fadd float %p18, %p19
  %v10 = fadd float %p20, %p21
  %v11 = fadd float %p22, %p23
  %v12 = fadd float %p24, %p25
  %v13 = fadd float %p26, %p27
  %v14 = fadd float %p28, %p29
  %v15 = fadd float %p30, %p31

  %v16 = fadd float %v00, %v01
  %v17 = fadd float %v02, %v03
  %v18 = fadd float %v04, %v05
  %v19 = fadd float %v06, %v07
  %v20 = fadd float %v08, %v09
  %v21 = fadd float %v10, %v11
  %v22 = fadd float %v12, %v13
  %v23 = fadd float %v14, %v15

  %v24 = fadd float %v16, %v17
  %v25 = fadd float %v18, %v19
  %v26 = fadd float %v20, %v21
  %v27 = fadd float %v22, %v23

  %v28 = fadd float %v24, %v25
  %v29 = fadd float %v26, %v27

  %v30 = fadd float %v28, %v29
  ret float %v30
}
attributes #0 = { "target-features"="+limit-16-fpregs" }

; Ensure that normally do use s0-s31 in the function above (if this isn't the
; case then we need to create a better test that forces usage of more fp regs).
; CHECK-LABEL: no_limit:
; CHECK-DAG: s0
; CHECK-DAG: s1
; CHECK-DAG: s2
; CHECK-DAG: s3
; CHECK-DAG: s4
; CHECK-DAG: s5
; CHECK-DAG: s6
; CHECK-DAG: s7
; CHECK-DAG: s8
; CHECK-DAG: s9
; CHECK-DAG: s10
; CHECK-DAG: s11
; CHECK-DAG: s12
; CHECK-DAG: s13
; CHECK-DAG: s14
; CHECK-DAG: s15
; CHECK-DAG: s16
; CHECK-DAG: s17
; CHECK-DAG: s18
; CHECK-DAG: s19
; CHECK-DAG: s20
; CHECK-DAG: s21
; CHECK-DAG: s22
; CHECK-DAG: s23
; CHECK-DAG: s24
; CHECK-DAG: s25
; CHECK-DAG: s26
; CHECK-DAG: s27
; CHECK-DAG: s28
; CHECK-DAG: s29
; CHECK-DAG: s30
; CHECK-DAG: s31
define float @no_limit(
  float %p00,
  float %p01,
  float %p02,
  float %p03,
  float %p04,
  float %p05,
  float %p06,
  float %p07,
  float %p08,
  float %p09,
  float %p10,
  float %p11,
  float %p12,
  float %p13,
  float %p14,
  float %p15,
  float %p16,
  float %p17,
  float %p18,
  float %p19,
  float %p20,
  float %p21,
  float %p22,
  float %p23,
  float %p24,
  float %p25,
  float %p26,
  float %p27,
  float %p28,
  float %p29,
  float %p30,
  float %p31
) {
  %v00 = fadd float %p00, %p01
  %v01 = fadd float %p02, %p03
  %v02 = fadd float %p04, %p05
  %v03 = fadd float %p06, %p07
  %v04 = fadd float %p08, %p09
  %v05 = fadd float %p10, %p11
  %v06 = fadd float %p12, %p13
  %v07 = fadd float %p14, %p15
  %v08 = fadd float %p16, %p17
  %v09 = fadd float %p18, %p19
  %v10 = fadd float %p20, %p21
  %v11 = fadd float %p22, %p23
  %v12 = fadd float %p24, %p25
  %v13 = fadd float %p26, %p27
  %v14 = fadd float %p28, %p29
  %v15 = fadd float %p30, %p31

  %v16 = fadd float %v00, %v01
  %v17 = fadd float %v02, %v03
  %v18 = fadd float %v04, %v05
  %v19 = fadd float %v06, %v07
  %v20 = fadd float %v08, %v09
  %v21 = fadd float %v10, %v11
  %v22 = fadd float %v12, %v13
  %v23 = fadd float %v14, %v15

  %v24 = fadd float %v16, %v17
  %v25 = fadd float %v18, %v19
  %v26 = fadd float %v20, %v21
  %v27 = fadd float %v22, %v23

  %v28 = fadd float %v24, %v25
  %v29 = fadd float %v26, %v27

  %v30 = fadd float %v28, %v29
  ret float %v30
}
