; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=stel

@x = external global float
@xd = external global double
@y = external global float
@yd = external global double
@ret_sf = external global float
@ret_df = external global double
@ret_sc = external global { float, float }
@ret_dc = external global { double, double }

; Function Attrs: nounwind
define void @v_sf(float %p) #0 {
entry:
  %p.addr = alloca float, align 4
  store float %p, float* %p.addr, align 4
  %0 = load float, float* %p.addr, align 4
  store float %0, float* @x, align 4
  ret void
}
; stel: .section .mips16.fn.v_sf,"ax",@progbits
; stel: .ent __fn_stub_v_sf
; stel: lui $25, %hi(v_sf)
; stel: addiu $25, $25, %lo(v_sf)
; stel: mfc1 $4, $f12
; stel: jr $25
; stel: __fn_local_v_sf = v_sf
; stel: .end __fn_stub_v_sf

declare i32 @printf(i8*, ...) #1

; Function Attrs: nounwind
define void @v_df(double %p) #0 {
entry:
  %p.addr = alloca double, align 8
  store double %p, double* %p.addr, align 8
  %0 = load double, double* %p.addr, align 8
  store double %0, double* @xd, align 8
  ret void
}

; stel: .section .mips16.fn.v_df,"ax",@progbits
; stel: .ent __fn_stub_v_df
; stel: lui $25, %hi(v_df)
; stel: addiu $25, $25, %lo(v_df)
; stel: mfc1 $4, $f12
; stel: mfc1 $5, $f13
; stel: jr $25
; stel: __fn_local_v_df = v_df
; stel: .end __fn_stub_v_df

; Function Attrs: nounwind
define void @v_sf_sf(float %p1, float %p2) #0 {
entry:
  %p1.addr = alloca float, align 4
  %p2.addr = alloca float, align 4
  store float %p1, float* %p1.addr, align 4
  store float %p2, float* %p2.addr, align 4
  %0 = load float, float* %p1.addr, align 4
  store float %0, float* @x, align 4
  %1 = load float, float* %p2.addr, align 4
  store float %1, float* @y, align 4
  ret void
}

; stel: .section .mips16.fn.v_sf_sf,"ax",@progbits
; stel: .ent __fn_stub_v_sf_sf
; stel: lui $25, %hi(v_sf_sf)
; stel: addiu $25, $25, %lo(v_sf_sf)
; stel: mfc1 $4, $f12
; stel: mfc1 $5, $f14
; stel: jr $25
; stel: __fn_local_v_sf_sf = v_sf_sf
; stel: .end __fn_stub_v_sf_sf

; Function Attrs: nounwind
define void @v_sf_df(float %p1, double %p2) #0 {
entry:
  %p1.addr = alloca float, align 4
  %p2.addr = alloca double, align 8
  store float %p1, float* %p1.addr, align 4
  store double %p2, double* %p2.addr, align 8
  %0 = load float, float* %p1.addr, align 4
  store float %0, float* @x, align 4
  %1 = load double, double* %p2.addr, align 8
  store double %1, double* @yd, align 8
  ret void
}

; stel: .section .mips16.fn.v_sf_df,"ax",@progbits
; stel: .ent __fn_stub_v_sf_df
; stel: lui $25, %hi(v_sf_df)
; stel: addiu $25, $25, %lo(v_sf_df)
; stel: mfc1 $4, $f12
; stel: mfc1 $6, $f14
; stel: mfc1 $7, $f15
; stel: jr $25
; stel: __fn_local_v_sf_df = v_sf_df
; stel: .end __fn_stub_v_sf_df

; Function Attrs: nounwind
define void @v_df_sf(double %p1, float %p2) #0 {
entry:
  %p1.addr = alloca double, align 8
  %p2.addr = alloca float, align 4
  store double %p1, double* %p1.addr, align 8
  store float %p2, float* %p2.addr, align 4
  %0 = load double, double* %p1.addr, align 8
  store double %0, double* @xd, align 8
  %1 = load float, float* %p2.addr, align 4
  store float %1, float* @y, align 4
  ret void
}

; stel: .section .mips16.fn.v_df_sf,"ax",@progbits
; stel: .ent __fn_stub_v_df_sf
; stel: lui $25, %hi(v_df_sf)
; stel: addiu $25, $25, %lo(v_df_sf)
; stel: mfc1 $4, $f12
; stel: mfc1 $5, $f13
; stel: mfc1 $6, $f14
; stel: jr $25
; stel: __fn_local_v_df_sf = v_df_sf
; stel: .end __fn_stub_v_df_sf

; Function Attrs: nounwind
define void @v_df_df(double %p1, double %p2) #0 {
entry:
  %p1.addr = alloca double, align 8
  %p2.addr = alloca double, align 8
  store double %p1, double* %p1.addr, align 8
  store double %p2, double* %p2.addr, align 8
  %0 = load double, double* %p1.addr, align 8
  store double %0, double* @xd, align 8
  %1 = load double, double* %p2.addr, align 8
  store double %1, double* @yd, align 8
  ret void
}

; stel: .section .mips16.fn.v_df_df,"ax",@progbits
; stel: .ent __fn_stub_v_df_df
; stel: lui $25, %hi(v_df_df)
; stel: addiu $25, $25, %lo(v_df_df)
; stel: mfc1 $4, $f12
; stel: mfc1 $5, $f13
; stel: mfc1 $6, $f14
; stel: mfc1 $7, $f15
; stel: jr $25
; stel: __fn_local_v_df_df = v_df_df
; stel: .end __fn_stub_v_df_df

; Function Attrs: nounwind
define float @sf_v() #0 {
entry:
  %0 = load float, float* @ret_sf, align 4
  ret float %0
}

; Function Attrs: nounwind
define float @sf_sf(float %p) #0 {
entry:
  %p.addr = alloca float, align 4
  store float %p, float* %p.addr, align 4
  %0 = load float, float* %p.addr, align 4
  store float %0, float* @x, align 4
  %1 = load float, float* @ret_sf, align 4
  ret float %1
}


; stel: .section .mips16.fn.sf_sf,"ax",@progbits
; stel: .ent __fn_stub_sf_sf
; stel: lui $25, %hi(sf_sf)
; stel: addiu $25, $25, %lo(sf_sf)
; stel: mfc1 $4, $f12
; stel: jr $25
; stel: __fn_local_sf_sf = sf_sf
; stel: .end __fn_stub_sf_sf


; Function Attrs: nounwind
define float @sf_df(double %p) #0 {
entry:
  %p.addr = alloca double, align 8
  store double %p, double* %p.addr, align 8
  %0 = load double, double* %p.addr, align 8
  store double %0, double* @xd, align 8
  %1 = load float, float* @ret_sf, align 4
  ret float %1
}

; stel: .section .mips16.fn.sf_df,"ax",@progbits
; stel: .ent __fn_stub_sf_df
; stel: lui $25, %hi(sf_df)
; stel: addiu $25, $25, %lo(sf_df)
; stel: mfc1 $4, $f12
; stel: mfc1 $5, $f13
; stel: jr $25
; stel: __fn_local_sf_df = sf_df
; stel: .end __fn_stub_sf_df

; Function Attrs: nounwind
define float @sf_sf_sf(float %p1, float %p2) #0 {
entry:
  %p1.addr = alloca float, align 4
  %p2.addr = alloca float, align 4
  store float %p1, float* %p1.addr, align 4
  store float %p2, float* %p2.addr, align 4
  %0 = load float, float* %p1.addr, align 4
  store float %0, float* @x, align 4
  %1 = load float, float* %p2.addr, align 4
  store float %1, float* @y, align 4
  %2 = load float, float* @ret_sf, align 4
  ret float %2
}

; stel: .section .mips16.fn.sf_sf_sf,"ax",@progbits
; stel: .ent __fn_stub_sf_sf_sf
; stel: lui $25, %hi(sf_sf_sf)
; stel: addiu $25, $25, %lo(sf_sf_sf)
; stel: mfc1 $4, $f12
; stel: mfc1 $5, $f14
; stel: jr $25
; stel: __fn_local_sf_sf_sf = sf_sf_sf
; stel: .end __fn_stub_sf_sf_sf

; Function Attrs: nounwind
define float @sf_sf_df(float %p1, double %p2) #0 {
entry:
  %p1.addr = alloca float, align 4
  %p2.addr = alloca double, align 8
  store float %p1, float* %p1.addr, align 4
  store double %p2, double* %p2.addr, align 8
  %0 = load float, float* %p1.addr, align 4
  store float %0, float* @x, align 4
  %1 = load double, double* %p2.addr, align 8
  store double %1, double* @yd, align 8
  %2 = load float, float* @ret_sf, align 4
  ret float %2
}

; stel: .section .mips16.fn.sf_sf_df,"ax",@progbits
; stel: .ent __fn_stub_sf_sf_df
; stel: lui $25, %hi(sf_sf_df)
; stel: addiu $25, $25, %lo(sf_sf_df)
; stel: mfc1 $4, $f12
; stel: mfc1 $6, $f14
; stel: mfc1 $7, $f15
; stel: jr $25
; stel: __fn_local_sf_sf_df = sf_sf_df
; stel: .end __fn_stub_sf_sf_df

; Function Attrs: nounwind
define float @sf_df_sf(double %p1, float %p2) #0 {
entry:
  %p1.addr = alloca double, align 8
  %p2.addr = alloca float, align 4
  store double %p1, double* %p1.addr, align 8
  store float %p2, float* %p2.addr, align 4
  %0 = load double, double* %p1.addr, align 8
  store double %0, double* @xd, align 8
  %1 = load float, float* %p2.addr, align 4
  store float %1, float* @y, align 4
  %2 = load float, float* @ret_sf, align 4
  ret float %2
}

; stel: .section .mips16.fn.sf_df_sf,"ax",@progbits
; stel: .ent __fn_stub_sf_df_sf
; stel: lui $25, %hi(sf_df_sf)
; stel: addiu $25, $25, %lo(sf_df_sf)
; stel: mfc1 $4, $f12
; stel: mfc1 $5, $f13
; stel: mfc1 $6, $f14
; stel: jr $25
; stel: __fn_local_sf_df_sf = sf_df_sf
; stel: .end __fn_stub_sf_df_sf

; Function Attrs: nounwind
define float @sf_df_df(double %p1, double %p2) #0 {
entry:
  %p1.addr = alloca double, align 8
  %p2.addr = alloca double, align 8
  store double %p1, double* %p1.addr, align 8
  store double %p2, double* %p2.addr, align 8
  %0 = load double, double* %p1.addr, align 8
  store double %0, double* @xd, align 8
  %1 = load double, double* %p2.addr, align 8
  store double %1, double* @yd, align 8
  %2 = load float, float* @ret_sf, align 4
  ret float %2
}

; stel: .section .mips16.fn.sf_df_df,"ax",@progbits
; stel: .ent __fn_stub_sf_df_df
; stel: lui $25, %hi(sf_df_df)
; stel: addiu $25, $25, %lo(sf_df_df)
; stel: mfc1 $4, $f12
; stel: mfc1 $5, $f13
; stel: mfc1 $6, $f14
; stel: mfc1 $7, $f15
; stel: jr $25
; stel: __fn_local_sf_df_df = sf_df_df
; stel: .end __fn_stub_sf_df_df

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
