; RUN: llc %s -O0 -march=sparc -mcpu=leon3 -mattr=+hasleoncasa -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -mcpu=gr712rc -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -mcpu=leon4 -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -mcpu=gr740 -o - | FileCheck %s

; CHECK-LABEL: casa_test
; CHECK:       casa [%o0] 10, %o3, %o2
define void @casa_test(i32* %ptr) {
  %pair = cmpxchg i32* %ptr, i32 0, i32 1 monotonic monotonic
  %r = extractvalue { i32, i1 } %pair, 0
  %stored1  = icmp eq i32 %r, 0

  ret void
}
