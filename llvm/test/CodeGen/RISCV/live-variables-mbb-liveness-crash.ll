; RUN: llc -mtriple=riscv64 -mattr=+rva22u64 -riscv-enable-live-variables -stop-after=riscv-live-variables,1 < %s
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

; CHECK:body:             |
; CHECK:  bb.0.bb:
; CHECK:    successors: %bb.1(0x00000000), %bb.2(0x80000000)
; CHECK:    EH_LABEL <mcsymbol >
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit killed $x2
; CHECK:    $x10 = COPY $x0
; CHECK:    $x11 = COPY $x0
; CHECK:    $x12 = COPY $x0
; CHECK:    PseudoCALL target-flags(riscv-call) @baz, csr_ilp32d_lp64d, implicit-def dead $x1, implicit killed $x10, implicit killed $x11, implicit killed $x12, implicit-def $x2
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit killed $x2
; CHECK:    EH_LABEL <mcsymbol >
; CHECK:    PseudoBR %bb.1
; CHECK:  bb.1.bb1:
; CHECK:    successors:
; CHECK:  bb.2.bb2 (landing-pad):
; CHECK:    EH_LABEL <mcsymbol >
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit killed $x2
; CHECK:    $x10 = COPY killed $x0
; CHECK:    PseudoCALL target-flags(riscv-call) @_Unwind_Resume, csr_ilp32d_lp64d, implicit-def dead $x1, implicit killed $x10, implicit-def $x2
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit killed $x2

define i1 @widget() personality ptr null {
bb:
  invoke void (ptr, ptr, ...) @baz(ptr null, ptr null, ptr null)
          to label %bb1 unwind label %bb2

bb1:                                              ; preds = %bb
  unreachable

bb2:                                              ; preds = %bb
  %landingpad = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } zeroinitializer
}

declare void @baz(ptr, ptr, ...)

