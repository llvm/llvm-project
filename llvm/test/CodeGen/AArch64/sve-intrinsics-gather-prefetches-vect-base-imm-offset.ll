; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; PRFB <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element
define void @llvm_aarch64_sve_prfb_gather_scalar_offset_nx4vi32(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scalar_offset_nx4vi32:
; CHECK-NEXT:  prfb  pldl1strm, p0, [z0.s, #7]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 7, i32 1)
  ret void
}

; PRFB <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element
define void @llvm_aarch64_sve_prfb_gather_scalar_offset_nx2vi64(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scalar_offset_nx2vi64:
; CHECK-NEXT:  prfb  pldl1strm, p0, [z0.d, #7]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 7, i32 1)
  ret void
}

; PRFH <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element
define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx4vi32:
; CHECK-NEXT:  prfh  pldl1strm, p0, [z0.s, #6]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 6, i32 1)
  ret void
}

; PRFH <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element
define void @llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scalar_offset_nx2vi64:
; CHECK-NEXT:  prfh  pldl1strm, p0, [z0.d, #6]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 6, i32 1)
  ret void
}

; PRFW <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element
define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx4vi32:
; CHECK-NEXT:  prfw  pldl1strm, p0, [z0.s, #12]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 12, i32 1)
  ret void
}

; PRFW <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element
define void @llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scalar_offset_nx2vi64:
; CHECK-NEXT:  prfw  pldl1strm, p0, [z0.d, #12]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 12, i32 1)
  ret void
}

; PRFD <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element
define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx4vi32:
; CHECK-NEXT:  prfd  pldl1strm, p0, [z0.s, #16]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 16, i32 1)
  ret void
}

; PRFD <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element
define void @llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scalar_offset_nx2vi64:
; CHECK-NEXT:  prfd  pldl1strm, p0, [z0.d, #16]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scalar.offset.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 16, i32 1)
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
