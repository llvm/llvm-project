; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; PRFB <prfop>, <Pg>, [<Xn|SP>, <Zm>.S, <mod>]    -> 32-bit indexes
define void @llvm_aarch64_sve_prfb_gather_uxtw_index_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_uxtw_index_nx4vi32:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.s, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.uxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfb_gather_scaled_sxtw_index_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scaled_sxtw_index_nx4vi32:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.s, sxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.sxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 1)
  ret void
 }

; PRFB <prfop>, <Pg>, [<Xn|SP>, <Zm>.D, <mod>]    -> 32-bit unpacked indexes

define void @llvm_aarch64_sve_prfb_gather_uxtw_index_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_uxtw_index_nx2vi64:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.d, uxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.uxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfb_gather_scaled_sxtw_index_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scaled_sxtw_index_nx2vi64:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.d, sxtw]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.sxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 1)
  ret void
 }
; PRFB <prfop>, <Pg>, [<Xn|SP>, <Zm>.D] -> 64-bit indexes
define void @llvm_aarch64_sve_prfb_gather_scaled_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfb_gather_scaled_nx2vi64:
; CHECK-NEXT:  prfb  pldl1strm, p0, [x0, z0.d]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfb.gather.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes, i32 1)
  ret void
 }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFH <prfop>, <Pg>, [<Xn|SP>, <Zm>.S, <mod>]    -> 32-bit indexes
define void @llvm_aarch64_sve_prfh_gather_uxtw_index_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_uxtw_index_nx4vi32:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.s, uxtw #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.uxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfh_gather_scaled_sxtw_index_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scaled_sxtw_index_nx4vi32:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.s, sxtw #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.sxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 1)
  ret void
 }

; PRFH <prfop>, <Pg>, [<Xn|SP>, <Zm>.D, <mod> #1] -> 32-bit unpacked indexes
define void @llvm_aarch64_sve_prfh_gather_uxtw_index_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_uxtw_index_nx2vi64:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.d, uxtw #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.uxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfh_gather_scaled_sxtw_index_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scaled_sxtw_index_nx2vi64:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.d, sxtw #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.sxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 1)
  ret void
 }

; PRFH <prfop>, <Pg>, [<Xn|SP>, <Zm>.D] -> 64-bit indexes
define void @llvm_aarch64_sve_prfh_gather_scaled_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfh_gather_scaled_nx2vi64:
; CHECK-NEXT:  prfh  pldl1strm, p0, [x0, z0.d, lsl #1]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfh.gather.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes, i32 1)
  ret void
 }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFW <prfop>, <Pg>, [<Xn|SP>, <Zm>.S, <mod>]    -> 32-bit indexes
define void @llvm_aarch64_sve_prfw_gather_uxtw_index_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_uxtw_index_nx4vi32:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.s, uxtw #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.uxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfw_gather_scaled_sxtw_index_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scaled_sxtw_index_nx4vi32:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.s, sxtw #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.sxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 1)
  ret void
 }

; PRFW <prfop>, <Pg>, [<Xn|SP>, <Zm>.D, <mod> #2] -> 32-bit unpacked indexes
define void @llvm_aarch64_sve_prfw_gather_uxtw_index_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_uxtw_index_nx2vi64:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.d, uxtw #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.uxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfw_gather_scaled_sxtw_index_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scaled_sxtw_index_nx2vi64:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.d, sxtw #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.sxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 1)
  ret void
 }

; PRFW <prfop>, <Pg>, [<Xn|SP>, <Zm>.D] -> 64-bit indexes
define void @llvm_aarch64_sve_prfw_gather_scaled_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfw_gather_scaled_nx2vi64:
; CHECK-NEXT:  prfw  pldl1strm, p0, [x0, z0.d, lsl #2]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfw.gather.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes, i32 1)
  ret void
 }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; PRFD <prfop>, <Pg>, [<Xn|SP>, <Zm>.S, <mod>]    -> 32-bit indexes
define void @llvm_aarch64_sve_prfd_gather_uxtw_index_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_uxtw_index_nx4vi32:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.s, uxtw #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.uxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfd_gather_scaled_sxtw_index_nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scaled_sxtw_index_nx4vi32:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.s, sxtw #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.sxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 1)
  ret void
 }

; PRFD <prfop>, <Pg>, [<Xn|SP>, <Zm>.D, <mod> #3] -> 32-bit unpacked indexes
define void @llvm_aarch64_sve_prfd_gather_uxtw_index_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_uxtw_index_nx2vi64:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.d, uxtw #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.uxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 1)
  ret void
 }

define void @llvm_aarch64_sve_prfd_gather_scaled_sxtw_index_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scaled_sxtw_index_nx2vi64:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.d, sxtw #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.sxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 1)
  ret void
 }

; PRFD <prfop>, <Pg>, [<Xn|SP>, <Zm>.D] -> 64-bit indexes
define void @llvm_aarch64_sve_prfd_gather_scaled_nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes) nounwind {
; CHECK-LABEL: llvm_aarch64_sve_prfd_gather_scaled_nx2vi64:
; CHECK-NEXT:  prfd  pldl1strm, p0, [x0, z0.d, lsl #3]
; CHECK-NEXT:  ret
  call void @llvm.aarch64.sve.prfd.gather.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes, i32 1)
  ret void
 }

declare void @llvm.aarch64.sve.prfb.gather.sxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.uxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.sxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.uxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfb.gather.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes, i32 %prfop)

declare void @llvm.aarch64.sve.prfh.gather.sxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.uxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.sxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.uxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfh.gather.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes, i32 %prfop)

declare void @llvm.aarch64.sve.prfw.gather.sxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.uxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.sxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.uxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfw.gather.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes, i32 %prfop)

declare void @llvm.aarch64.sve.prfd.gather.sxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.uxtw.index.nx4vi32(<vscale x 4 x i1> %Pg, i8* %base, <vscale x 4 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.sxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.uxtw.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i32> %indexes, i32 %prfop)
declare void @llvm.aarch64.sve.prfd.gather.index.nx2vi64(<vscale x 2 x i1> %Pg, i8* %base, <vscale x 2 x i64> %indexes, i32 %prfop)
