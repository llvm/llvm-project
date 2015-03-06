; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -relocation-model=static  < %s | FileCheck %s -check-prefix=static16

; Function Attrs: nounwind
define double @my_mul(double %a, double %b) #0 {
entry:
  %a.addr = alloca double, align 8
  %b.addr = alloca double, align 8
  store double %a, double* %a.addr, align 8
  store double %b, double* %b.addr, align 8
  %0 = load double* %a.addr, align 8
  %1 = load double* %b.addr, align 8
  %mul = fmul double %0, %1
  ret double %mul
}

; static16: 	        .ent	__fn_stub_my_mul
; static16:     	.set reorder
; static16-NEXT:	#NO_APP
; static16: 	        .end __fn_stub_my_mul
attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
