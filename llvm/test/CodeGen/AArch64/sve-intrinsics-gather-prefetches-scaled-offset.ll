; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s | FileCheck %s

; PRFB <prfop>, <Pg>, [<Xn|SP>, <Zm>.S, <mod>]    -> 32-bit          scaled offset
define void @llvm_aarch64_sve_prfb_gather_scaled_uxtw_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scaled_uxtw_nx4vi32:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scaled.uxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfb_gather_scaled_sxtw_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scaled_sxtw_nx4vi32:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.s, sxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scaled.sxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 1)
  ret void
 }

; PRFB <prfop>, <Pg>, [<Xn|SP>, <Zm>.D, <mod>]    -> 32-bit unpacked scaled offset

define void @llvm_aarch64_sve_prfb_gather_scaled_uxtw_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scaled_uxtw_nx2vi64:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scaled.uxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfb_gather_scaled_sxtw_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scaled_sxtw_nx2vi64:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.d, sxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scaled.sxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 1)
  ret void
 }
; PRFB <prfop>, <Pg>, [<Xn|SP>, <Zm>.D]           -> 64-bit          scaled offset
define void @llvm_aarch64_sve_prfb_gather_scaled_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scaled_nx2vi64:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.d]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.scaled.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset, i32 1)
  ret void
 }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFH <prfop>, <Pg>, [<Xn|SP>, <Zm>.S, <mod>]    -> 32-bit          scaled offset
define void @llvm_aarch64_sve_prfh_gather_scaled_uxtw_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scaled_uxtw_nx4vi32:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.s, uxtw #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scaled.uxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfh_gather_scaled_sxtw_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scaled_sxtw_nx4vi32:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.s, sxtw #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scaled.sxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 1)
  ret void
 }

; PRFH <prfop>, <Pg>, [<Xn|SP>, <Zm>.D, <mod> #1] -> 32-bit unpacked scaled offset
define void @llvm_aarch64_sve_prfh_gather_scaled_uxtw_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scaled_uxtw_nx2vi64:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.d, uxtw #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scaled.uxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfh_gather_scaled_sxtw_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scaled_sxtw_nx2vi64:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.d, sxtw #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scaled.sxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 1)
  ret void
 }

; PRFH <prfop>, <Pg>, [<Xn|SP>, <Zm>.D]           -> 64-bit          scaled offset
define void @llvm_aarch64_sve_prfh_gather_scaled_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scaled_nx2vi64:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.d, lsl #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.scaled.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset, i32 1)
  ret void
 }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFW <prfop>, <Pg>, [<Xn|SP>, <Zm>.S, <mod>]    -> 32-bit          scaled offset
define void @llvm_aarch64_sve_prfw_gather_scaled_uxtw_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scaled_uxtw_nx4vi32:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.s, uxtw #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scaled.uxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfw_gather_scaled_sxtw_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scaled_sxtw_nx4vi32:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.s, sxtw #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scaled.sxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 1)
  ret void
 }

; PRFW <prfop>, <Pg>, [<Xn|SP>, <Zm>.D, <mod> #2] -> 32-bit unpacked scaled offset
define void @llvm_aarch64_sve_prfw_gather_scaled_uxtw_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scaled_uxtw_nx2vi64:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.d, uxtw #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scaled.uxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfw_gather_scaled_sxtw_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scaled_sxtw_nx2vi64:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.d, sxtw #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scaled.sxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 1)
  ret void
 }

; PRFW <prfop>, <Pg>, [<Xn|SP>, <Zm>.D]           -> 64-bit          scaled offset
define void @llvm_aarch64_sve_prfw_gather_scaled_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scaled_nx2vi64:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.d, lsl #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.scaled.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset, i32 1)
  ret void
 }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFD <prfop>, <Pg>, [<Xn|SP>, <Zm>.S, <mod>]    -> 32-bit          scaled offset
define void @llvm_aarch64_sve_prfd_gather_scaled_uxtw_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scaled_uxtw_nx4vi32:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.s, uxtw #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scaled.uxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfd_gather_scaled_sxtw_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scaled_sxtw_nx4vi32:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.s, sxtw #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scaled.sxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 1)
  ret void
 }

; PRFD <prfop>, <Pg>, [<Xn|SP>, <Zm>.D, <mod> #3] -> 32-bit unpacked scaled offset
define void @llvm_aarch64_sve_prfd_gather_scaled_uxtw_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scaled_uxtw_nx2vi64:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.d, uxtw #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scaled.uxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfd_gather_scaled_sxtw_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scaled_sxtw_nx2vi64:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.d, sxtw #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scaled.sxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 1)
  ret void
 }

; PRFD <prfop>, <Pg>, [<Xn|SP>, <Zm>.D]           -> 64-bit          scaled offset
define void @llvm_aarch64_sve_prfd_gather_scaled_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scaled_nx2vi64:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.d, lsl #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.scaled.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset, i32 1)
  ret void
 }

declare void @llvm.aarch64.sve.prfb.gather.scaled.uxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.scaled.sxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.scaled.uxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.scaled.sxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.scaled.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.scaled.uxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.scaled.sxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.scaled.uxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.scaled.sxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.scaled.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.scaled.uxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.scaled.sxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.scaled.uxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.scaled.sxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.scaled.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.scaled.uxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.scaled.sxtw.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.scaled.uxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.scaled.sxtw.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %offset, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.scaled.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %offset, i32 %prfop)
