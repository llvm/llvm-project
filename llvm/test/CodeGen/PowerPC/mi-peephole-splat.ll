; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-linux-gnu < %s \
; RUN: | FileCheck --check-prefix=CHECK-LE %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-linux-gnu -mattr=+vsx < %s \
; RUN: | FileCheck --check-prefix=CHECK-BE %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-linux-gnu -mcpu=pwr9 < %s \
; RUN: | FileCheck --check-prefix=CHECK-P9LE %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-linux-gnu -mcpu=pwr9 < %s \
; RUN: | FileCheck --check-prefix=CHECK-P9BE %s

define double @splat_swap(<2 x double> %x, <2 x double> %y) nounwind  {
  %added = fadd <2 x double> %x, %y
  %call = tail call <2 x double> @llvm.rint.v2f64(<2 x double> %added) nounwind readnone
  %res1 = extractelement <2 x double> %call, i32 0
  %res2 = extractelement <2 x double> %call, i32 1
  %ret = fsub double %res1, %res2
  ret double %ret

; CHECK-LE-LABEL: splat_swap:
; CHECK-LE: xxmrghd [[XREG1:[0-9]+]], [[XREG1]], [[XREG2:[0-9]+]]
; CHECK-LE-NEXT: xxswapd [[XREG2]], [[XREG1]]
; CHECK-LE-NEXT: xssubdp [[XREG2]], [[XREG2]], [[XREG1]]
; CHECK-LE-NEXT: addi [[REG1:[0-9]+]], [[REG1]], {{[0-9]+}}
;
; CHECK-BE-LABEL: splat_swap:
; CHECK-BE: xxmrghd [[XREG1:[0-9]+]], [[XREG1]], [[XREG2:[0-9]+]]
; CHECK-BE-NEXT: xxswapd [[XREG2]], [[XREG1]]
; CHECK-BE-NEXT: xssubdp [[XREG2]], [[XREG1]], [[XREG2]]
; CHECK-BE-NEXT: addi [[REG1:[0-9]+]], [[REG1]], {{[0-9]+}}
;
; CHECK-P9LE-LABEL: splat_swap:
; CHECK-P9LE-DAG: xxmrghd [[XREG1:[0-9]+]], [[XREG1]], [[XREG2:[0-9]+]]
; CHECK-P9LE: xxswapd [[XREG2]], [[XREG1]]
; CHECK-P9LE-NEXT: xssubdp [[XREG2]], [[XREG2]], [[XREG1]]
; CHECK-P9LE-NEXT: addi [[REG1:[0-9]+]], [[REG1]], {{[0-9]+}}
;
; CHECK-P9BE-LABEL: splat_swap:
; CHECK-P9BE-DAG: xxmrghd [[XREG1:[0-9]+]], [[XREG1]], [[XREG2:[0-9]+]]
; CHECK-P9BE: xxswapd [[XREG2]], [[XREG1]]
; CHECK-P9BE-NEXT: xssubdp [[XREG2]], [[XREG1]], [[XREG2]]
; CHECK-P9BE-NEXT: addi [[REG1:[0-9]+]], [[REG1]], {{[0-9]+}}
}

declare <2 x double> @llvm.rint.v2f64(<2 x double>)

