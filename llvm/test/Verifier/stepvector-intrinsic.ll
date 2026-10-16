; RUN: not opt -S -passes=verify -disable-output < %s 2>&1 | FileCheck %s

; Reject vectors of integers less than 8 bits in width
; CHECK: stepvector only supported for vectors of integers with a bitwidth of at least 8
; CHECK-NEXT: call <vscale x 16 x i1> @llvm.stepvector.nxv16i1()
declare <vscale x 16 x i1> @llvm.stepvector.nxv16i1()
define <vscale x 16 x i1> @stepvector_i1() {
  %1 = call <vscale x 16 x i1> @llvm.stepvector.nxv16i1()
  ret <vscale x 16 x i1> %1
}
