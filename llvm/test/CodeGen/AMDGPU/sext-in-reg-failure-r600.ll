; RUN: llc -mtriple=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG %s
;
; EG-LABEL: {{^}}sext_in_reg_v2i1_in_v2i32_other_amount:
; EG: MEM_{{.*}} MSKOR [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG-NOT: BFE
; EG: ADD_INT
; EG: LSHL
; EG: ASHR
; EG: LSHL
; EG: ASHR
; EG: LSHR {{\*?}} [[ADDR]]

; Works with the align 2 removed
define amdgpu_kernel void @sext_in_reg_v2i1_in_v2i32_other_amount(ptr addrspace(1) %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b
  %x = shl <2 x i32> %c, <i32 6, i32 6>
  %y = ashr <2 x i32> %x, <i32 7, i32 7>
  store <2 x i32> %y, ptr addrspace(1) %out, align 2
  ret void
}
