; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false -xcoff-traceback-table=false < %s | FileCheck %s

; CHECK:      .extern .__tls_get_addr[PR]
; CHECK-NEXT: .csect .tdata[TL],2
; CHECK-NEXT: .globl  _Renamed..24f_f                 # @"f$f"
; CHECK-NEXT: .rename _Renamed..24f_f,"f$f"

@"f$f" = thread_local global i32 10, align 4

define void @fun() {
entry:
  store i32 1, ptr @"f$f", align 4
  ret void
}
