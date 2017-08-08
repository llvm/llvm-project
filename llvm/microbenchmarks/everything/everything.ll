; ModuleID = 'everything.c'
source_filename = "everything.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: noinline nounwind ssp uwtable
define i32 @foo() #0 {
entry:
  ret i32 10
}

; Function Attrs: noinline nounwind ssp uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %c = alloca double, align 8
  store i32 0, i32* %retval, align 4
  %call = call i32 @foo()
  %conv = sitofp i32 %call to double
  store double %conv, double* %c, align 8
  detach label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %0 = bitcast i32 undef to i32
  %1 = load double, double* %c, align 8
  %call1 = call double @sin(double %1) #2
  %2 = load double, double* %c, align 8
  %add = fadd double %2, %call1
  store double %add, double* %c, align 8
  %3 = load double, double* %c, align 8
  %call2 = call double @sin(double %3) #2
  %4 = load double, double* %c, align 8
  %add3 = fadd double %4, %call2
  store double %add3, double* %c, align 8
  %5 = load double, double* %c, align 8
  %call4 = call double @sin(double %5) #2
  %6 = load double, double* %c, align 8
  %add5 = fadd double %6, %call4
  store double %add5, double* %c, align 8
  reattach label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  detach label %det.achd6, label %det.cont15

det.achd6:                                        ; preds = %det.cont
  %7 = bitcast i32 undef to i32
  detach label %det.achd7, label %det.cont14

det.achd7:                                        ; preds = %det.achd6
  %8 = bitcast i32 undef to i32
  %9 = load double, double* %c, align 8
  %call8 = call double @sin(double %9) #2
  %10 = load double, double* %c, align 8
  %add9 = fadd double %10, %call8
  store double %add9, double* %c, align 8
  %11 = load double, double* %c, align 8
  %call10 = call double @sin(double %11) #2
  %12 = load double, double* %c, align 8
  %add11 = fadd double %12, %call10
  store double %add11, double* %c, align 8
  %13 = load double, double* %c, align 8
  %call12 = call double @sin(double %13) #2
  %14 = load double, double* %c, align 8
  %add13 = fadd double %14, %call12
  store double %add13, double* %c, align 8
  reattach label %det.cont14

det.cont14:                                       ; preds = %det.achd7, %det.achd6
  reattach label %det.cont15

det.cont15:                                       ; preds = %det.cont14, %det.cont
  detach label %det.achd16, label %det.cont23

det.achd16:                                       ; preds = %det.cont15
  %15 = bitcast i32 undef to i32
  %16 = load double, double* %c, align 8
  %tobool = fcmp une double %16, 0.000000e+00
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %det.achd16
  %17 = load double, double* %c, align 8
  %call17 = call double @sin(double %17) #2
  %18 = load double, double* %c, align 8
  %add18 = fadd double %18, %call17
  store double %add18, double* %c, align 8
  %19 = load double, double* %c, align 8
  %call19 = call double @sin(double %19) #2
  %20 = load double, double* %c, align 8
  %add20 = fadd double %20, %call19
  store double %add20, double* %c, align 8
  %21 = load double, double* %c, align 8
  %call21 = call double @sin(double %21) #2
  %22 = load double, double* %c, align 8
  %add22 = fadd double %22, %call21
  store double %add22, double* %c, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %det.achd16
  reattach label %det.cont23

det.cont23:                                       ; preds = %if.end, %det.cont15
  %23 = load double, double* %c, align 8
  %conv24 = fptosi double %23 to i32
  ret i32 %conv24
}

; Function Attrs: nounwind readnone
declare double @sin(double) #1

attributes #0 = { noinline nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 5.0.0 (git@github.com:wsmoses/Cilk-Clang 5942594810265567795884c83b5a37a8cbc98d3e) (git@github.com:wsmoses/Parallel-IR 8f57e0739bf9fc6736472c89f91a533630efd5c3)"}
