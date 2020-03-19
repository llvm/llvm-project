; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s | FileCheck %s

; PRFB <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element
define void @llvm_aarch64_sve_gather_prfb_nx4vi32(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_gather_prfb_nx4vi32:
; CHECK-NEXT:  prfb  pldl1strm, p0, [z0.s, #7]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.gather.prfb.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 7, i32 1)
  ret void
}

; PRFB <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element
define void @llvm_aarch64_sve_gather_prfb_nx2vi64(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_gather_prfb_nx2vi64:
; CHECK-NEXT:  prfb  pldl1strm, p0, [z0.d, #7]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.gather.prfb.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 7, i32 1)
  ret void
}

; PRFH <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element
define void @llvm_aarch64_sve_gather_prfh_nx4vi32(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_gather_prfh_nx4vi32:
; CHECK-NEXT:  prfh  pldl1strm, p0, [z0.s, #6]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.gather.prfh.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 6, i32 1)
  ret void
}

; PRFH <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element
define void @llvm_aarch64_sve_gather_prfh_nx2vi64(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_gather_prfh_nx2vi64:
; CHECK-NEXT:  prfh  pldl1strm, p0, [z0.d, #6]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.gather.prfh.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 6, i32 1)
  ret void
}

; PRFW <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element
define void @llvm_aarch64_sve_gather_prfw_nx4vi32(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_gather_prfw_nx4vi32:
; CHECK-NEXT:  prfw  pldl1strm, p0, [z0.s, #12]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.gather.prfw.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 12, i32 1)
  ret void
}

; PRFW <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element
define void @llvm_aarch64_sve_gather_prfw_nx2vi64(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_gather_prfw_nx2vi64:
; CHECK-NEXT:  prfw  pldl1strm, p0, [z0.d, #12]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.gather.prfw.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 12, i32 1)
  ret void
}

; PRFD <prfop>, <Pg>, [<Zn>.S{, #<imm>}] -> 32-bit element
define void @llvm_aarch64_sve_gather_prfd_nx4vi32(<vscale x 4 x i32> %bases, <vscale x 4 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_gather_prfd_nx4vi32:
; CHECK-NEXT:  prfd  pldl1strm, p0, [z0.s, #16]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.gather.prfd.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 16, i32 1)
  ret void
}

; PRFD <prfop>, <Pg>, [<Zn>.D{, #<imm>}] -> 64-bit element
define void @llvm_aarch64_sve_gather_prfd_nx2vi64(<vscale x 2 x i64> %bases, <vscale x 2 x i1> %Pg) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_gather_prfd_nx2vi64:
; CHECK-NEXT:  prfd  pldl1strm, p0, [z0.d, #16]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.gather.prfd.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 16, i32 1)
  ret void
}

declare void @llvm.aarch64.sve.gather.prfb.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %imm, i32 %prfop)
declare void @llvm.aarch64.sve.gather.prfb.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %imm, i32 %prfop)
declare void @llvm.aarch64.sve.gather.prfh.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %imm, i32 %prfop)
declare void @llvm.aarch64.sve.gather.prfh.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %imm, i32 %prfop)
declare void @llvm.aarch64.sve.gather.prfw.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %imm, i32 %prfop)
declare void @llvm.aarch64.sve.gather.prfw.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %imm, i32 %prfop)
declare void @llvm.aarch64.sve.gather.prfd.nx4vi32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> %bases, i64 %imm, i32 %prfop)
declare void @llvm.aarch64.sve.gather.prfd.nx2vi64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> %bases, i64 %imm, i32 %prfop)
