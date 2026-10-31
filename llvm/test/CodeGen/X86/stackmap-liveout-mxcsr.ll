; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -stop-after=stackmap-liveness -o - | FileCheck %s
;
; Check that a register without a DWARF register number is not recorded in
; the live-out list of a patchpoint. MXCSR is implicitly read by UCOMISD and
; has no DWARF number, so the AsmPrinter used to trip the assert in
; getDwarfRegNum when emitting the patchpoint's stackmap.

define i1 @f(double %val) {
entry:
  %p = call ptr addrspace(1) (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.p1(i64 0, i32 0, ptr null, i32 0)
  %c = fcmp uno double %val, 0.000000e+00
  ret i1 %c
}

declare ptr addrspace(1) @llvm.experimental.patchpoint.p1(i64 immarg, i32 immarg, ptr, i32 immarg, ...)

; CHECK: PATCHPOINT {{.*}}liveout(
; CHECK-NOT: $mxcsr
; CHECK: UCOMISD
