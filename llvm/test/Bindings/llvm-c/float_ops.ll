; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
source_filename = "/test/Bindings/float_ops.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"


define float @float_ops_f32(float %a, float %b) {
  %1 = fneg float %a

  %2 = fadd float %a, %b
  %3 = fsub float %a, %b
  %4 = fmul float %a, %b
  %5 = fdiv float %a, %b
  %6 = frem float %a, %b

  ret float %1
}

define double @float_ops_f64(double %a, double %b) {
  %1 = fneg double %a

  %2 = fadd double %a, %b
  %3 = fsub double %a, %b
  %4 = fmul double %a, %b
  %5 = fdiv double %a, %b
  %6 = frem double %a, %b

  ret double %1
}

define void @float_cmp_f32(float %a, float %b) {
  %1  = fcmp oeq float %a, %b
  %2  = fcmp ogt float %a, %b
  %3  = fcmp olt float %a, %b
  %4  = fcmp ole float %a, %b
  %5  = fcmp one float %a, %b

  %6  = fcmp ueq float %a, %b
  %7  = fcmp ugt float %a, %b
  %8  = fcmp ult float %a, %b
  %9  = fcmp ule float %a, %b
  %10 = fcmp une float %a, %b

  %11 = fcmp ord float %a, %b
  %12 = fcmp false float %a, %b
  %13 = fcmp true float %a, %b

  ret void
}

define void @float_cmp_f64(double %a, double %b) {
  %1  = fcmp oeq double %a, %b
  %2  = fcmp ogt double %a, %b
  %3  = fcmp olt double %a, %b
  %4  = fcmp ole double %a, %b
  %5  = fcmp one double %a, %b

  %6  = fcmp ueq double %a, %b
  %7  = fcmp ugt double %a, %b
  %8  = fcmp ult double %a, %b
  %9  = fcmp ule double %a, %b
  %10 = fcmp une double %a, %b

  %11 = fcmp ord double %a, %b
  %12 = fcmp false double %a, %b
  %13 = fcmp true double %a, %b

  ret void
}

define void @float_cmp_fast_f32(float %a, float %b) {
  %1  = fcmp fast oeq float %a, %b
  %2  = fcmp nsz ogt float %a, %b
  %3  = fcmp nsz nnan olt float %a, %b
  %4  = fcmp contract ole float %a, %b
  %5  = fcmp nnan one float %a, %b

  %6  = fcmp nnan ninf nsz ueq float %a, %b
  %7  = fcmp arcp ugt float %a, %b
  %8  = fcmp fast ult float %a, %b
  %9  = fcmp fast ule float %a, %b
  %10 = fcmp fast une float %a, %b

  %11 = fcmp fast ord float %a, %b
  %12 = fcmp nnan ninf false float %a, %b
  %13 = fcmp nnan ninf true float %a, %b

  ret void
}

define void @float_cmp_fast_f64(double %a, double %b) {
  %1  = fcmp fast oeq double %a, %b
  %2  = fcmp nsz ogt double %a, %b
  %3  = fcmp nsz nnan olt double %a, %b
  %4  = fcmp contract ole double %a, %b
  %5  = fcmp nnan one double %a, %b

  %6  = fcmp nnan ninf nsz ueq double %a, %b
  %7  = fcmp arcp ugt double %a, %b
  %8  = fcmp fast ult double %a, %b
  %9  = fcmp fast ule double %a, %b
  %10 = fcmp fast une double %a, %b

  %11 = fcmp fast ord double %a, %b
  %12 = fcmp nnan ninf false double %a, %b
  %13 = fcmp nnan ninf true double %a, %b

  ret void
}

define float @float_ops_fast_f32(float %a, float %b) {
  %1 = fneg nnan float %a

  %2 = fadd ninf float %a, %b
  %3 = fsub nsz float %a, %b
  %4 = fmul arcp float %a, %b
  %5 = fdiv contract float %a, %b
  %6 = frem afn float %a, %b

  %7 = fadd reassoc float %a, %b
  %8 = fadd reassoc float %7, %b

  %9  = fadd fast float %a, %b
  %10 = fadd nnan nsz float %a, %b
  %11 = frem nnan nsz float %a, %b
  %12 = fdiv nnan nsz arcp float %a, %b
  %13 = fmul nnan nsz ninf contract float %a, %b
  %14 = fmul nnan nsz ninf arcp contract afn reassoc float %a, %b

  ret float %1
}

define double @float_ops_fast_f64(double %a, double %b) {
  %1 = fneg nnan double %a

  %2 = fadd ninf double %a, %b
  %3 = fsub nsz double %a, %b
  %4 = fmul arcp double %a, %b
  %5 = fdiv contract double %a, %b
  %6 = frem afn double %a, %b

  %7 = fadd reassoc double %a, %b
  %8 = fadd reassoc double %7, %b

  %9  = fadd fast double %a, %b
  %10 = fadd nnan nsz double %a, %b
  %11 = frem nnan nsz double %a, %b
  %12 = fdiv nnan nsz arcp double %a, %b
  %13 = fmul nnan nsz ninf contract double %a, %b
  %14 = fmul nnan nsz ninf arcp contract afn reassoc double %a, %b

  ret double %1
}

