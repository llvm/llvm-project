; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s | FileCheck %s

; PRFB <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element, imm = 0, 1, ..., 31
define void @llvm_aarch64_sve_prfb_gather_scalar_offset_nx4vi32_runtime_offset(<vscale x 4 x i32> %bases, i64 %offset, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scalar_offset_nx4vi32_runtime_offset:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %offset, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfb_gather_scalar_offset_nx4vi32_invalid_immediate_offset_upper_bound(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scalar_offset_nx4vi32_invalid_immediate_offset_upper_bound:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #32
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 32, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfb_gather_scalar_offset_nx4vi32_invalid_immediate_offset_lower_bound(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scalar_offset_nx4vi32_invalid_immediate_offset_lower_bound:
; CHECK-NEXT:  mov   x[[N:[0-9]+]], #-1
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 -1, i32 1)
  ret void
}

; PRFB <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element, imm = 0, 1, ..., 31
define void @llvm_aarch64_sve_prfb_gather_scalar_offset_nx2vi64_runtime_offset(<vscale x 2 x i64> %bases, i64 %offset, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scalar_offset_nx2vi64_runtime_offset:
; CHECK-NEXT:   prfb pldl1strm, p0, [x0, z0.d, uxtw]
; CHECK-NEXT:   ret
  call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %offset, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfb_gather_scalar_offset_nx2vi64_invalid_immediate_offset_upper_bound(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scalar_offset_nx2vi64_invalid_immediate_offset_upper_bound:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #32
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 32, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfb_gather_scalar_offset_nx2vi64_invalid_immediate_offset_lower_bound(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scalar_offset_nx2vi64_invalid_immediate_offset_lower_bound:
; CHECK-NEXT:  mov   x[[N:[0-9]+]], #-1
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 -1, i32 1)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFH <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element, imm = 0, 2, ..., 62
define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32_runtime_offset(<vscale x 4 x i32> %bases, i64 %offset, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32_runtime_offset:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %offset, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32_invalid_immediate_offset_upper_bound(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32_invalid_immediate_offset_upper_bound:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #63
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 63, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32_invalid_immediate_offset_lower_bound(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32_invalid_immediate_offset_lower_bound:
; CHECK-NEXT:  mov   x[[N:[0-9]+]], #-1
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 -1, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32_invalid_immediate_offset_inbound_not_multiple_of_2(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32_invalid_immediate_offset_inbound_not_multiple_of_2:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #33
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 33, i32 1)
  ret void
}

; PRFH <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element, imm = 0, 2, ..., 62
define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64_runtime_offset(<vscale x 2 x i64> %bases, i64 %offset, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64_runtime_offset:
; CHECK-NEXT:  prfb pldl1strm, p0, [x0, z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %offset, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64_invalid_immediate_offset_upper_bound(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64_invalid_immediate_offset_upper_bound:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #63
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 63, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64_invalid_immediate_offset_lower_bound(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64_invalid_immediate_offset_lower_bound:
; CHECK-NEXT:  mov   x[[N:[0-9]+]], #-1
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 -1, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64_invalid_immediate_offset_inbound_not_multiple_of_2(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64_invalid_immediate_offset_inbound_not_multiple_of_2:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #33
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 33, i32 1)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFW <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element, imm = 0, 4, ..., 124
define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32_runtime_offset(<vscale x 4 x i32> %bases, i64 %offset, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32_runtime_offset:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %offset, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32_invalid_immediate_offset_upper_bound(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32_invalid_immediate_offset_upper_bound:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #125
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 125, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32_invalid_immediate_offset_lower_bound(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32_invalid_immediate_offset_lower_bound:
; CHECK-NEXT:  mov   x[[N:[0-9]+]], #-1
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 -1, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32_invalid_immediate_offset_inbound_not_multiple_of_4(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32_invalid_immediate_offset_inbound_not_multiple_of_4:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #33
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 33, i32 1)
  ret void
}

; PRFW <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element, imm = 0, 4, ..., 124
define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64_runtime_offset(<vscale x 2 x i64> %bases, i64 %offset, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64_runtime_offset:
; CHECK-NEXT:   prfb pldl1strm, p0, [x0, z0.d, uxtw]
; CHECK-NEXT:   ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %offset, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64_invalid_immediate_offset_upper_bound(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64_invalid_immediate_offset_upper_bound:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #125
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 125, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64_invalid_immediate_offset_lower_bound(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64_invalid_immediate_offset_lower_bound:
; CHECK-NEXT:  mov   x[[N:[0-9]+]], #-1
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 -1, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64_invalid_immediate_offset_inbound_not_multiple_of_4(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64_invalid_immediate_offset_inbound_not_multiple_of_4:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #33
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 33, i32 1)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFD <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element, imm = 0, 8, ..., 248
define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32_runtime_offset(<vscale x 4 x i32> %bases, i64 %offset, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32_runtime_offset:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %offset, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32_invalid_immediate_offset_upper_bound(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32_invalid_immediate_offset_upper_bound:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #125
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 125, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32_invalid_immediate_offset_lower_bound(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32_invalid_immediate_offset_lower_bound:
; CHECK-NEXT:  mov   x[[N:[0-9]+]], #-1
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 -1, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32_invalid_immediate_offset_inbound_not_multiple_of_8(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32_invalid_immediate_offset_inbound_not_multiple_of_8:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #33
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 33, i32 1)
  ret void
}

; PRFD <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element, imm = 0, 4, ..., 248
define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64_runtime_offset(<vscale x 2 x i64> %bases, i64 %offset, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64_runtime_offset:
; CHECK-NEXT:   prfb pldl1strm, p0, [x0, z0.d, uxtw]
; CHECK-NEXT:   ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %offset, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64_invalid_immediate_offset_upper_bound(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64_invalid_immediate_offset_upper_bound:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #125
; CHECK-NEXT:  prfb pldl1strm, p0, [x[[N]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 125, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64_invalid_immediate_offset_lower_bound(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64_invalid_immediate_offset_lower_bound:
; CHECK-NEXT:  mov   x[[N:[0-9]+]], #-1
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 -1, i32 1)
  ret void
}

define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64_invalid_immediate_offset_inbound_not_multiple_of_8(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64_invalid_immediate_offset_inbound_not_multiple_of_8:
; CHECK-NEXT:  mov   w[[N:[0-9]+]], #33
; CHECK-NEXT:  prfb  pldl1strm, p0, [x[[N:[0-9]+]], z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 33, i32 1)
  ret void
}

declare void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %offset, i32 %prfop)
