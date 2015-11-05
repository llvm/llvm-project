; RUN: llc -march=mips -mcpu=mips32r2 -mattr=+micromips,+nooddspreg -O0 < %s | FileCheck %s

; CHECK: addiu    $[[R0:[0-9]+]], $zero, 0
; CHECK: subu16   $2, $[[R0]], ${{[0-9]+}}
define i32 @foo() {
  %1 = sub i32 0, undef
  ret i32 %1
}
