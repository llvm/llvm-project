; RUN: llc -verify-machineinstrs -mcpu=pwr9  < %s -mtriple=powerpc64le-unknown-linux-gnu -mattr=-power9-vector | FileCheck -check-prefixes=CHECK-PWR8,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr8  < %s -mtriple=powerpc64le-unknown-linux-gnu -mattr=+power9-vector | FileCheck -check-prefixes=CHECK-PWR9,CHECK-ALL %s 

declare <2 x double> @__cbrtd2_massv(<2 x double>)
declare <4 x float> @__cbrtf4_massv(<4 x float>)

; cbrt without the power9-vector attribute on the caller
; check massv calls are correctly targeted for Power8
define <2 x double>  @cbrt_f64_massv_nopwr9(<2 x double> %opnd) #0 {
; CHECK-ALL-LABEL: @cbrt_f64_massv_nopwr9
; CHECK-PWR8: bl __cbrtd2_P8
; CHECK-NOT: bl __cbrtd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__cbrtd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

; cbrt with the power9-vector attribute on the caller
; check massv calls are correctly targeted for Power9
define <2 x double>  @cbrt_f64_massv_pwr9(<2 x double> %opnd) #1 {
; CHECK-ALL-LABEL: @cbrt_f64_massv_pwr9
; CHECK-PWR9: bl __cbrtd2_P9
; CHECK-NOT: bl __cbrtd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__cbrtd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}
