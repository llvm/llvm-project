; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 %s -o - | FileCheck %s

; bug 15283
; radar://13191881
; CHECK: vfcmp
define void @vfcmp(ptr %a, ptr %b) {
  %wide.load = load <2 x double>, ptr %a, align 4
  %wide.load2 = load <2 x double>, ptr %b, align 4
; CHECK-NOT: vdup.32
; CHECK-NOT: vmovn.i64
  %v1 = fcmp olt <2 x double> %wide.load, %wide.load2
  %v2 = zext <2 x i1> %v1 to <2 x i32>
  %v3 = sitofp <2 x i32> %v2 to <2 x double>
  store <2 x double> %v3, ptr %b, align 4
  ret void
}
