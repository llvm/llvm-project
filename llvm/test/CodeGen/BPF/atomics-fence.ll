; RUN: llc -mtriple=bpfel -mcpu=v3 < %s | FileCheck %s
; RUN: not --crash llc -mtriple=bpfel -mcpu=v2 < %s 2>&1 | FileCheck --check-prefix=V2 %s
;
; An atomic insn that returns the old value is fully ordered. v1 and v2 have
; 'lock *(u64 *)(r10 - 8) += r1' only, which is not.

define void @fence() {
; CHECK-LABEL: fence:
; CHECK:       r{{[0-9]}} = atomic_fetch_add((u64 *)(r10 - 8), r{{[0-9]}})
; V2:          Runtime fence is not supported at the moment
  fence seq_cst
  ret void
}
