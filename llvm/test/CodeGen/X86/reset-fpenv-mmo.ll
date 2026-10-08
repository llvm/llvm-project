; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel < %s | FileCheck %s

; LowerRESET_FPENV builds a MachineMemOperand for the constant-pool blob it
; loads via FLDENVm. The flag must be MOLoad: FLDENVm is mayLoad = 1, so a
; MOStore-flagged MMO is silently dropped by SelectionDAGISel's memref
; filter, leaving FLDENVm with no memrefs at all. Verify the load-direction
; MMO survives to the final MachineInstr.

declare void @llvm.reset.fpenv()

define void @reset_fpenv_mmo() nounwind {
  ; CHECK-LABEL: name: reset_fpenv_mmo
  ; CHECK: FLDENVm {{.*}} :: (load (s224) from constant-pool, align 4)
  ; CHECK-NEXT: LDMXCSR {{.*}} implicit-def dead $mxcsr
  ; CHECK-NEXT: RET 0
  call void @llvm.reset.fpenv()
  ret void
}
