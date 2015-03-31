; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=R600 -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}rotr_i32:
; R600: BIT_ALIGN_INT

; SI: v_alignbit_b32
define void @rotr_i32(i32 addrspace(1)* %in, i32 %x, i32 %y) {
entry:
  %tmp0 = sub i32 32, %y
  %tmp1 = shl i32 %x, %tmp0
  %tmp2 = lshr i32 %x, %y
  %tmp3 = or i32 %tmp1, %tmp2
  store i32 %tmp3, i32 addrspace(1)* %in
  ret void
}

; FUNC-LABEL: {{^}}rotr_v2i32:
; R600: BIT_ALIGN_INT
; R600: BIT_ALIGN_INT

; SI: v_alignbit_b32
; SI: v_alignbit_b32
define void @rotr_v2i32(<2 x i32> addrspace(1)* %in, <2 x i32> %x, <2 x i32> %y) {
entry:
  %tmp0 = sub <2 x i32> <i32 32, i32 32>, %y
  %tmp1 = shl <2 x i32> %x, %tmp0
  %tmp2 = lshr <2 x i32> %x, %y
  %tmp3 = or <2 x i32> %tmp1, %tmp2
  store <2 x i32> %tmp3, <2 x i32> addrspace(1)* %in
  ret void
}

; FUNC-LABEL: {{^}}rotr_v4i32:
; R600: BIT_ALIGN_INT
; R600: BIT_ALIGN_INT
; R600: BIT_ALIGN_INT
; R600: BIT_ALIGN_INT

; SI: v_alignbit_b32
; SI: v_alignbit_b32
; SI: v_alignbit_b32
; SI: v_alignbit_b32
define void @rotr_v4i32(<4 x i32> addrspace(1)* %in, <4 x i32> %x, <4 x i32> %y) {
entry:
  %tmp0 = sub <4 x i32> <i32 32, i32 32, i32 32, i32 32>, %y
  %tmp1 = shl <4 x i32> %x, %tmp0
  %tmp2 = lshr <4 x i32> %x, %y
  %tmp3 = or <4 x i32> %tmp1, %tmp2
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %in
  ret void
}
