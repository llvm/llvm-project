; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32r6 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mips-linux-gnu -mcpu=mips32r6 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32r6 -target-abi=o32 -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=mips-linux-gnu -mcpu=mips32r6 -target-abi=o32 -verify-machineinstrs -filetype=obj < %s -o /dev/null

; A condition shared by two scalarized SEL.D instructions reaches one use via
; an FPR copy. copyPhysReg must not require an immediately visible SEL.D use
; when moving this condition from a GPR32 to an FGR64 register.
define void @shared_condition(ptr %out, <2 x double> %a, <2 x double> %b, i1 %c) {
; CHECK-LABEL: shared_condition:
; CHECK: mtc1
; CHECK: sel.d
; CHECK: sel.d
; CHECK: cmp.lt.d
; CHECK: sel.d
  %selected = select i1 %c, <2 x double> %b, <2 x double> %a
  %cmp = fcmp olt <2 x double> %a, %selected
  %result = select <2 x i1> %cmp, <2 x double> %a, <2 x double> %selected
  store <2 x double> %result, ptr %out, align 16
  ret void
}
