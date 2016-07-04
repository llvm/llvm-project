; XFAIL: *
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s
; XUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG %s
;
; EG-LABEL: {{^}}sext_in_reg_v2i1_in_v2i32_other_amount:
; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG-NOT: BFE
; EG: ADD_INT
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]

; Works with the align 2 removed
define void @sext_in_reg_v2i1_in_v2i32_other_amount(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b
  %x = shl <2 x i32> %c, <i32 6, i32 6>
  %y = ashr <2 x i32> %x, <i32 7, i32 7>
  store <2 x i32> %y, <2 x i32> addrspace(1)* %out, align 2
  ret void
}
