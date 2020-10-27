; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define <vscale x 16 x i1> @vselect_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: vselect_16:
; CHECK: sel p0.b, p0, p1.b, p2.b
; CHECK-NEXT: ret
  %res = select <vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @vselect_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: vselect_8:
; CHECK: sel p0.b, p0, p1.b, p2.b
; CHECK-NEXT: ret
  %res = select <vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @vselect_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: vselect_4:
; CHECK: sel p0.b, p0, p1.b, p2.b
; CHECK-NEXT: ret
  %res = select <vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @vselect_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: vselect_2:
; CHECK: sel p0.b, p0, p1.b, p2.b
; CHECK-NEXT: ret
  %res = select <vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @and_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: and_16:
; CHECK: and p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @and_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: and_8:
; CHECK: and p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.and.z.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @and_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: and_4:
; CHECK: and p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.and.z.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @and_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: and_2:
; CHECK: and p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.and.z.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @bic_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: bic_16:
; CHECK: bic p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.bic.z.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @bic_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: bic_8:
; CHECK: bic p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.bic.z.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @bic_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: bic_4:
; CHECK: bic p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.bic.z.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @bic_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: bic_2:
; CHECK: bic p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.bic.z.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @eor_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: eor_16:
; CHECK: eor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.eor.z.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @eor_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: eor_8:
; CHECK: eor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.eor.z.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @eor_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: eor_4:
; CHECK: eor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.eor.z.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @eor_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: eor_2:
; CHECK: eor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.eor.z.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @orr_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: orr_16:
; CHECK: orr p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.orr.z.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @orr_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: orr_8:
; CHECK: orr p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.orr.z.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @orr_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: orr_4:
; CHECK: orr p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.orr.z.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @orr_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: orr_2:
; CHECK: orr p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.orr.z.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @orn_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: orn_16:
; CHECK: orn p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.orn.z.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @orn_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: orn_8:
; CHECK: orn p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.orn.z.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @orn_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: orn_4:
; CHECK: orn p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.orn.z.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @orn_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: orn_2:
; CHECK: orn p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.orn.z.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @nor_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: nor_16:
; CHECK: nor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.nor.z.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @nor_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: nor_8:
; CHECK: nor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.nor.z.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @nor_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: nor_4:
; CHECK: nor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.nor.z.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @nor_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: nor_2:
; CHECK: nor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.nor.z.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @nand_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: nand_16:
; CHECK: nand p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.nand.z.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn,  <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @nand_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: nand_8:
; CHECK: nand p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.nand.z.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @nand_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: nand_4:
; CHECK: nand p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.nand.z.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @nand_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: nand_2:
; CHECK: nand p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.nand.z.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.and.z.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.and.z.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.and.z.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.bic.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.bic.z.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.bic.z.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.bic.z.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.eor.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.eor.z.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.eor.z.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.eor.z.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.orr.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.orr.z.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.orr.z.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.orr.z.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.orn.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.orn.z.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.orn.z.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.orn.z.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.nor.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.nor.z.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.nor.z.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.nor.z.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.nand.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.nand.z.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.nand.z.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.nand.z.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
