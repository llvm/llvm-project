; RUN: llc < %s -march=xcore | FileCheck %s
define void @_Z1fz(...) {
entry:
; CHECK: _Z1fz:
; CHECK: extsp 3
; CHECK: stw r[[REG:[0-3]{1,1}]]
; CHECK: , sp{{\[}}[[REG]]{{\]}}
; CHECK: stw r[[REG:[0-3]{1,1}]]
; CHECK: , sp{{\[}}[[REG]]{{\]}}
; CHECK: stw r[[REG:[0-3]{1,1}]]
; CHECK: , sp{{\[}}[[REG]]{{\]}}
; CHECK: stw r[[REG:[0-3]{1,1}]]
; CHECK: , sp{{\[}}[[REG]]{{\]}}
; CHECK: ldaw sp, sp[3]
; CHECK: retsp 0
  ret void
}
