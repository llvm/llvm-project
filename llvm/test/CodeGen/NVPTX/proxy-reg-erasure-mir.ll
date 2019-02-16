; RUN: llc -march=nvptx64 -stop-before=nvptx-proxyreg-erasure < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MIR --check-prefix=MIR-BEFORE

; RUN: llc -march=nvptx64 -stop-after=nvptx-proxyreg-erasure < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MIR --check-prefix=MIR-AFTER

; Check ProxyRegErasure pass MIR manipulation.

declare <4 x i32> @callee_vec_i32()
define  <4 x i32> @check_vec_i32() {
  ; MIR: body:
  ; MIR-DAG: Callseq_Start {{[0-9]+}}, {{[0-9]+}}
  ; MIR-DAG: %0:int32regs, %1:int32regs, %2:int32regs, %3:int32regs = LoadParamMemV4I32 0
  ; MIR-DAG: Callseq_End {{[0-9]+}}

  ; MIR-BEFORE-DAG: %4:int32regs = ProxyRegI32 killed %0
  ; MIR-BEFORE-DAG: %5:int32regs = ProxyRegI32 killed %1
  ; MIR-BEFORE-DAG: %6:int32regs = ProxyRegI32 killed %2
  ; MIR-BEFORE-DAG: %7:int32regs = ProxyRegI32 killed %3
  ; MIR-BEFORE-DAG: StoreRetvalV4I32 killed %4, killed %5, killed %6, killed %7, 0
  ; MIR-AFTER-DAG:  StoreRetvalV4I32 killed %0, killed %1, killed %2, killed %3, 0

  %ret = call <4 x i32> @callee_vec_i32()
  ret <4 x i32> %ret
}
