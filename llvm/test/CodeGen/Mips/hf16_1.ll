; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=1
; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=2


@x = common global float 0.000000e+00, align 4
@xd = common global double 0.000000e+00, align 8
@y = common global float 0.000000e+00, align 4
@yd = common global double 0.000000e+00, align 8
@xy = common global { float, float } zeroinitializer, align 4
@xyd = common global { double, double } zeroinitializer, align 8

define void @foo() nounwind {
entry:
  %0 = load float, ptr @x, align 4
  call void @v_sf(float %0)
  %1 = load double, ptr @xd, align 8
  call void @v_df(double %1)
  %2 = load float, ptr @x, align 4
  %3 = load float, ptr @y, align 4
  call void @v_sf_sf(float %2, float %3)
  %4 = load double, ptr @xd, align 8
  %5 = load float, ptr @x, align 4
  call void @v_df_sf(double %4, float %5)
  %6 = load double, ptr @xd, align 8
  %7 = load double, ptr @yd, align 8
  call void @v_df_df(double %6, double %7)
  %call = call float @sf_v()
  %8 = load float, ptr @x, align 4
  %call1 = call float @sf_sf(float %8)
  %9 = load double, ptr @xd, align 8
  %call2 = call float @sf_df(double %9)
  %10 = load float, ptr @x, align 4
  %11 = load float, ptr @y, align 4
  %call3 = call float @sf_sf_sf(float %10, float %11)
  %12 = load double, ptr @xd, align 8
  %13 = load float, ptr @x, align 4
  %call4 = call float @sf_df_sf(double %12, float %13)
  %14 = load double, ptr @xd, align 8
  %15 = load double, ptr @yd, align 8
  %call5 = call float @sf_df_df(double %14, double %15)
  %call6 = call double @df_v()
  %16 = load float, ptr @x, align 4
  %call7 = call double @df_sf(float %16)
  %17 = load double, ptr @xd, align 8
  %call8 = call double @df_df(double %17)
  %18 = load float, ptr @x, align 4
  %19 = load float, ptr @y, align 4
  %call9 = call double @df_sf_sf(float %18, float %19)
  %20 = load double, ptr @xd, align 8
  %21 = load float, ptr @x, align 4
  %call10 = call double @df_df_sf(double %20, float %21)
  %22 = load double, ptr @xd, align 8
  %23 = load double, ptr @yd, align 8
  %call11 = call double @df_df_df(double %22, double %23)
  %call12 = call { float, float } @sc_v()
  %24 = extractvalue { float, float } %call12, 0
  %25 = extractvalue { float, float } %call12, 1
  %26 = load float, ptr @x, align 4
  %call13 = call { float, float } @sc_sf(float %26)
  %27 = extractvalue { float, float } %call13, 0
  %28 = extractvalue { float, float } %call13, 1
  %29 = load double, ptr @xd, align 8
  %call14 = call { float, float } @sc_df(double %29)
  %30 = extractvalue { float, float } %call14, 0
  %31 = extractvalue { float, float } %call14, 1
  %32 = load float, ptr @x, align 4
  %33 = load float, ptr @y, align 4
  %call15 = call { float, float } @sc_sf_sf(float %32, float %33)
  %34 = extractvalue { float, float } %call15, 0
  %35 = extractvalue { float, float } %call15, 1
  %36 = load double, ptr @xd, align 8
  %37 = load float, ptr @x, align 4
  %call16 = call { float, float } @sc_df_sf(double %36, float %37)
  %38 = extractvalue { float, float } %call16, 0
  %39 = extractvalue { float, float } %call16, 1
  %40 = load double, ptr @xd, align 8
  %41 = load double, ptr @yd, align 8
  %call17 = call { float, float } @sc_df_df(double %40, double %41)
  %42 = extractvalue { float, float } %call17, 0
  %43 = extractvalue { float, float } %call17, 1
  %call18 = call { double, double } @dc_v()
  %44 = extractvalue { double, double } %call18, 0
  %45 = extractvalue { double, double } %call18, 1
  %46 = load float, ptr @x, align 4
  %call19 = call { double, double } @dc_sf(float %46)
  %47 = extractvalue { double, double } %call19, 0
  %48 = extractvalue { double, double } %call19, 1
  %49 = load double, ptr @xd, align 8
  %call20 = call { double, double } @dc_df(double %49)
  %50 = extractvalue { double, double } %call20, 0
  %51 = extractvalue { double, double } %call20, 1
  %52 = load float, ptr @x, align 4
  %53 = load float, ptr @y, align 4
  %call21 = call { double, double } @dc_sf_sf(float %52, float %53)
  %54 = extractvalue { double, double } %call21, 0
  %55 = extractvalue { double, double } %call21, 1
  %56 = load double, ptr @xd, align 8
  %57 = load float, ptr @x, align 4
  %call22 = call { double, double } @dc_df_sf(double %56, float %57)
  %58 = extractvalue { double, double } %call22, 0
  %59 = extractvalue { double, double } %call22, 1
  %60 = load double, ptr @xd, align 8
  %61 = load double, ptr @yd, align 8
  %call23 = call { double, double } @dc_df_df(double %60, double %61)
  %62 = extractvalue { double, double } %call23, 0
  %63 = extractvalue { double, double } %call23, 1
  ret void
}

declare void @v_sf(float)

declare void @v_df(double)

declare void @v_sf_sf(float, float)

declare void @v_df_sf(double, float)

declare void @v_df_df(double, double)

declare float @sf_v()

declare float @sf_sf(float)

declare float @sf_df(double)

declare float @sf_sf_sf(float, float)

declare float @sf_df_sf(double, float)

declare float @sf_df_df(double, double)

declare double @df_v()

declare double @df_sf(float)

declare double @df_df(double)

declare double @df_sf_sf(float, float)

declare double @df_df_sf(double, float)

declare double @df_df_df(double, double)

declare { float, float } @sc_v()

declare { float, float } @sc_sf(float)

declare { float, float } @sc_df(double)

declare { float, float } @sc_sf_sf(float, float)

declare { float, float } @sc_df_sf(double, float)

declare { float, float } @sc_df_df(double, double)

declare { double, double } @dc_v()

declare { double, double } @dc_sf(float)

declare { double, double } @dc_df(double)

declare { double, double } @dc_sf_sf(float, float)

declare { double, double } @dc_df_sf(double, float)

declare { double, double } @dc_df_df(double, double)

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_1)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(v_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_2)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(v_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_5)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(v_sf_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_6)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(v_df_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_10)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(v_df_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_0)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sf_v)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sf_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_2)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sf_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_5)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sf_sf_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_6)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sf_df_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_10)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sf_df_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_0)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(df_v)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_1)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(df_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(df_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_5)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(df_sf_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_6)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(df_df_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_10)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(df_df_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sc_0)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sc_v)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sc_1)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sc_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sc_2)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sc_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sc_5)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sc_sf_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sc_6)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sc_df_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sc_10)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(sc_df_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_dc_0)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(dc_v)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_dc_1)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(dc_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_dc_2)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(dc_df)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_dc_5)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(dc_sf_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_dc_6)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(dc_df_sf)(${{[0-9]+}})

; 1:	lw	${{[0-9]+}}, %got(__mips16_call_stub_dc_10)(${{[0-9]+}})
; 2:	lw	${{[0-9]+}}, %call16(dc_df_df)(${{[0-9]+}})



