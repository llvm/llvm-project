; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}


; CHECK: bfe0
define i32 @bfe0(i32 %a) {
; CHECK: bfe.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, 4, 4
; CHECK-NOT: shr
; CHECK-NOT: and
  %val0 = ashr i32 %a, 4
  %val1 = and i32 %val0, 15
  ret i32 %val1
}

; CHECK: bfe1
define i32 @bfe1(i32 %a) {
; CHECK: bfe.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, 3, 3
; CHECK-NOT: shr
; CHECK-NOT: and
  %val0 = ashr i32 %a, 3
  %val1 = and i32 %val0, 7
  ret i32 %val1
}

; CHECK: bfe2
define i32 @bfe2(i32 %a) {
; CHECK: bfe.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, 5, 3
; CHECK-NOT: shr
; CHECK-NOT: and
  %val0 = ashr i32 %a, 5
  %val1 = and i32 %val0, 7
  ret i32 %val1
}
