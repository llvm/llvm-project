; RUN: llc %s -O0 -march=sparc -mcpu=leon2 -mattr=+filldatacache -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -mcpu=at697e -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -mcpu=at697f -mattr=+filldatacache -o - | FileCheck %s

; CHECK-LABEL: test_filldatacache_1
; CHECK:       or %g0, 1, %g1
; CHECK:       nop
; CHECK-NEXT:  add %g1, 1, %g1
; CHECK-NEXT:  cmp %g1, 4096
; CHECK-NEXT:  ble {{.+}}
define zeroext i1@test_filldatacache_1(i1 zeroext %a, i1 zeroext %b) {
  %1 = tail call zeroext i1 asm sideeffect "udivcc $0, $1, $2", "=r,r,r"(i1 zeroext %a, i1 zeroext %b)

  ret i1 %1
}


; CHECK-LABEL: test_filldatacache_2
; CHECK-NOT:   or %g0, 1, %g1
; CHECK-NOT:   add %g1, 1, %g1
; CHECK-NOT:   cmp %g1, 4096
; CHECK-NOT:   ble {{.+}}
define zeroext i1@test_filldatacache_2(i1 zeroext %a, i1 zeroext %b) {
  %1 = tail call zeroext i1 asm sideeffect "sdivcc $0, $1, $2", "=r,r,r"(i1 zeroext %a, i1 zeroext %b)

  ret i1 %1
}
