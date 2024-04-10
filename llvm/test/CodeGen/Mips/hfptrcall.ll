; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=picel

@ptrsv = global ptr @sv, align 4
@ptrdv = global ptr @dv, align 4
@ptrscv = global ptr @scv, align 4
@ptrdcv = global ptr @dcv, align 4
@x = common global float 0.000000e+00, align 4
@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@xd = common global double 0.000000e+00, align 8
@xy = common global { float, float } zeroinitializer, align 4
@.str1 = private unnamed_addr constant [10 x i8] c"%f + %fi\0A\00", align 1
@xyd = common global { double, double } zeroinitializer, align 8

; Function Attrs: nounwind
define float @sv() #0 {
entry:
  ret float 1.000000e+01
}
; picel: 	.ent	sv
; picel: 	lw	${{[0-9]+}}, %call16(__mips16_ret_sf)(${{[0-9]+}})
; picel:	.end	sv

; Function Attrs: nounwind
define double @dv() #0 {
entry:
  ret double 1.500000e+01
}

; picel: 	.ent	dv
; picel: 	lw	${{[0-9]+}}, %call16(__mips16_ret_df)(${{[0-9]+}})
; picel:	.end	dv

; Function Attrs: nounwind
define { float, float } @scv() #0 {
entry:
  %retval = alloca { float, float }, align 4
  %real = getelementptr inbounds { float, float }, ptr %retval, i32 0, i32 0
  %imag = getelementptr inbounds { float, float }, ptr %retval, i32 0, i32 1
  store float 5.000000e+00, ptr %real
  store float 9.900000e+01, ptr %imag
  %0 = load { float, float }, ptr %retval
  ret { float, float } %0
}

; picel: 	.ent	scv
; picel: 	lw	${{[0-9]+}}, %call16(__mips16_ret_sc)(${{[0-9]+}})
; picel:	.end	scv

; Function Attrs: nounwind
define { double, double } @dcv() #0 {
entry:
  %retval = alloca { double, double }, align 8
  %real = getelementptr inbounds { double, double }, ptr %retval, i32 0, i32 0
  %imag = getelementptr inbounds { double, double }, ptr %retval, i32 0, i32 1
  store double 0x416BC8B0A0000000, ptr %real
  store double 0x41CDCCB763800000, ptr %imag
  %0 = load { double, double }, ptr %retval
  ret { double, double } %0
}

; picel: 	.ent	dcv
; picel: 	lw	${{[0-9]+}}, %call16(__mips16_ret_dc)(${{[0-9]+}})
; picel:	.end	dcv

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %0 = load ptr, ptr @ptrsv, align 4
  %call = call float %0()
  store float %call, ptr @x, align 4
  %1 = load float, ptr @x, align 4
  %conv = fpext float %1 to double
  %call1 = call i32 (ptr, ...) @printf(ptr @.str, double %conv)
  %2 = load ptr, ptr @ptrdv, align 4
  %call2 = call double %2()
  store double %call2, ptr @xd, align 8
  %3 = load double, ptr @xd, align 8
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, double %3)
  %4 = load ptr, ptr @ptrscv, align 4
  %call4 = call { float, float } %4()
  %5 = extractvalue { float, float } %call4, 0
  %6 = extractvalue { float, float } %call4, 1
  store float %5, ptr @xy
  store float %6, ptr getelementptr inbounds ({ float, float }, ptr @xy, i32 0, i32 1)
  %xy.real = load float, ptr @xy
  %xy.imag = load float, ptr getelementptr inbounds ({ float, float }, ptr @xy, i32 0, i32 1)
  %conv5 = fpext float %xy.real to double
  %conv6 = fpext float %xy.imag to double
  %xy.real7 = load float, ptr @xy
  %xy.imag8 = load float, ptr getelementptr inbounds ({ float, float }, ptr @xy, i32 0, i32 1)
  %conv9 = fpext float %xy.real7 to double
  %conv10 = fpext float %xy.imag8 to double
  %call11 = call i32 (ptr, ...) @printf(ptr @.str1, double %conv5, double %conv10)
  %7 = load ptr, ptr @ptrdcv, align 4
  %call12 = call { double, double } %7()
  %8 = extractvalue { double, double } %call12, 0
  %9 = extractvalue { double, double } %call12, 1
  store double %8, ptr @xyd
  store double %9, ptr getelementptr inbounds ({ double, double }, ptr @xyd, i32 0, i32 1)
  %xyd.real = load double, ptr @xyd
  %xyd.imag = load double, ptr getelementptr inbounds ({ double, double }, ptr @xyd, i32 0, i32 1)
  %xyd.real13 = load double, ptr @xyd
  %xyd.imag14 = load double, ptr getelementptr inbounds ({ double, double }, ptr @xyd, i32 0, i32 1)
  %call15 = call i32 (ptr, ...) @printf(ptr @.str1, double %xyd.real, double %xyd.imag14)
  ret i32 0
}

; picel: 	.ent	main

; picel:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_0)(${{[0-9]+}})

; picel:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_0)(${{[0-9]+}})

; picel:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sc_0)(${{[0-9]+}})

; picel:	lw	${{[0-9]+}}, %got(__mips16_call_stub_dc_0)(${{[0-9]+}})


declare i32 @printf(ptr, ...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }



