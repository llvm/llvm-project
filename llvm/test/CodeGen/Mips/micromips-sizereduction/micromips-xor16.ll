; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: nounwind readnone
define i1 @fun(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: fun:
; CHECK: xor16
  %reg1 = or i32 %a, %b
  %reg2 = xor i32 %reg1, -1
  %bool1 = icmp ne i32 %a, -1
  %bool1.ext = zext i1 %bool1 to i32
  %bool2 = icmp eq i32 %bool1.ext, %reg2
  ret i1 %bool2
}
