; RUN: llc -mtriple=wasm32-unknown-unknown -asm-verbose=false < %s | FileCheck %s --check-prefix=CHECKCG
; RUN: llc -mtriple=wasm32-unknown-unknown -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=CHECKISEL

%f32_cell = type ptr addrspace(1)

; CHECKISEL-LABEL: name: ir_local_f32
; CHECKISEL:       stack:
; CHECKISEL:       id: 0, name: retval, type: default, offset: 1, size: 1, alignment: 4,
; CHECKISEL-NEXT:  stack-id: wasm-local

; CHECKCG-LABEL: ir_local_f32:
; CHECKCG-NEXT: .functype ir_local_f32 (f32) -> (f32)
; CHECKCG-NEXT: .local f32
; CHECKCG-NEXT: local.get 0
; CHECKCG-NEXT: local.set 1

define float @ir_local_f32(float %arg) {
 %retval = alloca float, addrspace(1)
 store float %arg, %f32_cell %retval
 %reloaded = load float, %f32_cell %retval
 ret float %reloaded
}
