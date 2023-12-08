; RUN: llc < %s -mcpu=swift -mtriple=armv7s-apple-ios | FileCheck %s
; RUN: llc < %s -arm-assume-misaligned-load-store -mcpu=swift -mtriple=armv7s-apple-ios | FileCheck %s

; Check that we avoid producing vldm instructions using d registers that
; begin in the most-significant half of a q register. These require more
; micro-ops on swift and so aren't worth combining.

; CHECK-LABEL: test_vldm
; CHECK: vldmia r{{[0-9]+}}, {d2, d3, d4}
; CHECK-NOT: vldmia r{{[0-9]+}}, {d1, d2, d3, d4}

declare fastcc void @force_register(double %d0, double %d1, double %d2, double %d3, double %d4) 

define void @test_vldm(ptr %x, ptr %y) {
entry:
  %addr1 = getelementptr double, ptr %x, i32 1
  %addr2 = getelementptr double, ptr %x, i32 2
  %addr3 = getelementptr double, ptr %x, i32 3
  %d0 = load double , ptr %y
  %d1 = load double , ptr %x
  %d2 = load double , ptr %addr1
  %d3 = load double , ptr %addr2
  %d4 = load double , ptr %addr3
  ; We are trying to force x[0-3] in registers d1 to d4 so that we can test we
  ; don't form a "vldmia rX, {d1, d2, d3, d4}".
  ; We are relying on the calling convention and that register allocation
  ; properly coalesces registers.
  call fastcc void @force_register(double %d0, double %d1, double %d2, double %d3, double %d4)
  ret void
}
