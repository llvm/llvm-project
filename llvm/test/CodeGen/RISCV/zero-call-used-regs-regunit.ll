; RUN: llc < %s -mtriple=riscv64-unknown-linux-gnu -mattr=+v -verify-machineinstrs | FileCheck %s

; Returning a vector in v8 must not reset the unrelated physical register
; whose MCRegister ID equals RegUnit(V8) (RISCV::V7), leaving v7 unzeroed.
define <vscale x 2 x i32> @ret_v8_uses_v7(ptr %p) nounwind "zero-call-used-regs"="used" {
; CHECK-LABEL: ret_v8_uses_v7:
; CHECK:       vl1re32.v v7, (a0)
; CHECK:       vs1r.v v7, (a0)
; CHECK:       vmv.v.i v7, 0
; CHECK:       ret
entry:
  %v0  = load volatile <vscale x 2 x i32>, ptr %p
  %v1  = load volatile <vscale x 2 x i32>, ptr %p
  %v2  = load volatile <vscale x 2 x i32>, ptr %p
  %v3  = load volatile <vscale x 2 x i32>, ptr %p
  %v4  = load volatile <vscale x 2 x i32>, ptr %p
  %v5  = load volatile <vscale x 2 x i32>, ptr %p
  %v6  = load volatile <vscale x 2 x i32>, ptr %p
  %v7  = load volatile <vscale x 2 x i32>, ptr %p
  %v8  = load volatile <vscale x 2 x i32>, ptr %p
  %v9  = load volatile <vscale x 2 x i32>, ptr %p
  %v10 = load volatile <vscale x 2 x i32>, ptr %p
  %v11 = load volatile <vscale x 2 x i32>, ptr %p
  %v12 = load volatile <vscale x 2 x i32>, ptr %p
  %v13 = load volatile <vscale x 2 x i32>, ptr %p
  %v14 = load volatile <vscale x 2 x i32>, ptr %p
  %v15 = load volatile <vscale x 2 x i32>, ptr %p
  %v16 = load volatile <vscale x 2 x i32>, ptr %p
  %v17 = load volatile <vscale x 2 x i32>, ptr %p
  %v18 = load volatile <vscale x 2 x i32>, ptr %p
  %v19 = load volatile <vscale x 2 x i32>, ptr %p
  %v20 = load volatile <vscale x 2 x i32>, ptr %p
  %v21 = load volatile <vscale x 2 x i32>, ptr %p
  %v22 = load volatile <vscale x 2 x i32>, ptr %p
  %v23 = load volatile <vscale x 2 x i32>, ptr %p
  %v24 = load volatile <vscale x 2 x i32>, ptr %p
  %v25 = load volatile <vscale x 2 x i32>, ptr %p
  store volatile <vscale x 2 x i32> %v0, ptr %p
  store volatile <vscale x 2 x i32> %v1, ptr %p
  store volatile <vscale x 2 x i32> %v2, ptr %p
  store volatile <vscale x 2 x i32> %v3, ptr %p
  store volatile <vscale x 2 x i32> %v4, ptr %p
  store volatile <vscale x 2 x i32> %v5, ptr %p
  store volatile <vscale x 2 x i32> %v6, ptr %p
  store volatile <vscale x 2 x i32> %v7, ptr %p
  store volatile <vscale x 2 x i32> %v8, ptr %p
  store volatile <vscale x 2 x i32> %v9, ptr %p
  store volatile <vscale x 2 x i32> %v10, ptr %p
  store volatile <vscale x 2 x i32> %v11, ptr %p
  store volatile <vscale x 2 x i32> %v12, ptr %p
  store volatile <vscale x 2 x i32> %v13, ptr %p
  store volatile <vscale x 2 x i32> %v14, ptr %p
  store volatile <vscale x 2 x i32> %v15, ptr %p
  store volatile <vscale x 2 x i32> %v16, ptr %p
  store volatile <vscale x 2 x i32> %v17, ptr %p
  store volatile <vscale x 2 x i32> %v18, ptr %p
  store volatile <vscale x 2 x i32> %v19, ptr %p
  store volatile <vscale x 2 x i32> %v20, ptr %p
  store volatile <vscale x 2 x i32> %v21, ptr %p
  store volatile <vscale x 2 x i32> %v22, ptr %p
  store volatile <vscale x 2 x i32> %v23, ptr %p
  store volatile <vscale x 2 x i32> %v24, ptr %p
  ret <vscale x 2 x i32> %v25
}
