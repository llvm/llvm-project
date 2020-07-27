; RUN: llc -verify-machineinstrs -mcpu=a2q < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-bgq-linux"

; Function Attrs: nounwind
define void @gsl_sf_legendre_Pl_deriv_array(<4 x i32> %inp1, <4 x double> %inp2) #0 {
entry:
  br label %vector.body198

vector.body198:                                   ; preds = %vector.body198, %for.body46.lr.ph
  %0 = icmp ne <4 x i32> %inp1, zeroinitializer
  %1 = select <4 x i1> %0, <4 x double> <double 5.000000e-01, double 5.000000e-01, double 5.000000e-01, double 5.000000e-01>, <4 x double> <double -5.000000e-01, double -5.000000e-01, double -5.000000e-01, double -5.000000e-01>
  %2 = fmul <4 x double> %inp2, %1
  %3 = fmul <4 x double> %inp2, %2
  %4 = fmul <4 x double> %3, %inp2
  store <4 x double> %4, <4 x double>* undef, align 8
  br label %return

; CHECK-LABEL: @gsl_sf_legendre_Pl_deriv_array
; CHECK: qvlfiwzx
; CHECK: qvfcfidu
; CHECK: qvfcmpeq
; CHECK: qvfsel
; CHECK: qvfmul

return:                                           ; preds = %if.else.i
  ret void
}

attributes #0 = { nounwind }

