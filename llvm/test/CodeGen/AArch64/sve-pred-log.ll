; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

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
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.and.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @and_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: and_8:
; CHECK: and p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.and.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @and_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: and_4:
; CHECK: and p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.and.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @and_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: and_2:
; CHECK: and p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.and.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}


define <vscale x 16 x i1> @bic_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: bic_16:
; CHECK: bic p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.bic.pred.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @bic_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: bic_8:
; CHECK: bic p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.bic.pred.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @bic_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: bic_4:
; CHECK: bic p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.bic.pred.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @bic_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: bic_2:
; CHECK: bic p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.bic.pred.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @eor_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: eor_16:
; CHECK: eor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.eor.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @eor_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: eor_8:
; CHECK: eor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.eor.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @eor_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: eor_4:
; CHECK: eor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.eor.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @eor_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: eor_2:
; CHECK: eor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.eor.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @ands_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: ands_16:
; CHECK: ands p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.ands.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @ands_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: ands_8:
; CHECK: ands p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.ands.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @ands_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: ands_4:
; CHECK: ands p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.ands.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @ands_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: ands_2:
; CHECK: ands p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.ands.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}


define <vscale x 16 x i1> @bics_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: bics_16:
; CHECK: bics p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.bics.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @bics_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: bics_8:
; CHECK: bics p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.bics.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @bics_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: bics_4:
; CHECK: bics p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.bics.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @bics_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: bics_2:
; CHECK: bics p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.bics.nxv2i1(<vscale x 2 x i1> %Pg,
    <vscale x 2 x i1> %Pn,
    <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}


define <vscale x 16 x i1> @eors_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: eors_16:
; CHECK: eors p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.eors.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @eors_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: eors_8:
; CHECK: eors p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.eors.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @eors_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: eors_4:
; CHECK: eors p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.eors.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @eors_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: eors_2:
; CHECK: eors p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.eors.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}


define <vscale x 16 x i1> @orr_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: orr_16:
; CHECK: orr p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.orr.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @orr_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: orr_8:
; CHECK: orr p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.orr.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @orr_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: orr_4:
; CHECK: orr p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.orr.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @orr_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: orr_2:
; CHECK: orr p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.orr.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}


define <vscale x 16 x i1> @orn_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: orn_16:
; CHECK: orn p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.orn.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @orn_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: orn_8:
; CHECK: orn p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.orn.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @orn_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: orn_4:
; CHECK: orn p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.orn.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @orn_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: orn_2:
; CHECK: orn p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.orn.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @nor_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: nor_16:
; CHECK: nor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.nor.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @nor_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: nor_8:
; CHECK: nor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.nor.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @nor_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: nor_4:
; CHECK: nor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.nor.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @nor_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: nor_2:
; CHECK: nor p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.nor.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @nand_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: nand_16:
; CHECK: nand p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.nand.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn,  <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @nand_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: nand_8:
; CHECK: nand p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.nand.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @nand_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: nand_4:
; CHECK: nand p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.nand.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @nand_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: nand_2:
; CHECK: nand p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.nand.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @orrs_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: orrs_16:
; CHECK: orrs p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.orrs.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @orrs_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: orrs_8:
; CHECK: orrs p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.orrs.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @orrs_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: orrs_4:
; CHECK: orrs p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.orrs.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @orrs_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: orrs_2:
; CHECK: orrs p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.orrs.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @orns_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: orns_16:
; CHECK: orns p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.orns.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @orns_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: orns_8:
; CHECK: orns p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.orns.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @orns_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: orns_4:
; CHECK: orns p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.orns.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn,  <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @orns_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: orns_2:
; CHECK: orns p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.orns.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @nors_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: nors_16:
; CHECK: nors p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.nors.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @nors_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: nors_8:
; CHECK: nors p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.nors.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @nors_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: nors_4:
; CHECK: nors p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.nors.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @nors_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: nors_2:
; CHECK: nors p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.nors.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

define <vscale x 16 x i1> @nands_16(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd) {
; CHECK-LABEL: nands_16:
; CHECK: nands p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.nands.nxv16i1(<vscale x 16 x i1> %Pg, <vscale x 16 x i1> %Pn, <vscale x 16 x i1> %Pd)
  ret <vscale x 16 x i1> %res;
}

define <vscale x 8 x i1> @nands_8(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd) {
; CHECK-LABEL: nands_8:
; CHECK: nands p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.nands.nxv8i1(<vscale x 8 x i1> %Pg, <vscale x 8 x i1> %Pn, <vscale x 8 x i1> %Pd)
  ret <vscale x 8 x i1> %res;
}

define <vscale x 4 x i1> @nands_4(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd) {
; CHECK-LABEL: nands_4:
; CHECK: nands p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.nands.nxv4i1(<vscale x 4 x i1> %Pg, <vscale x 4 x i1> %Pn, <vscale x 4 x i1> %Pd)
  ret <vscale x 4 x i1> %res;
}

define <vscale x 2 x i1> @nands_2(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd) {
; CHECK-LABEL: nands_2:
; CHECK: nands p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.nands.nxv2i1(<vscale x 2 x i1> %Pg, <vscale x 2 x i1> %Pn, <vscale x 2 x i1> %Pd)
  ret <vscale x 2 x i1> %res;
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.and.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.and.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.and.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.and.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.bic.pred.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.bic.pred.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.bic.pred.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.bic.pred.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.eor.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.eor.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.eor.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.eor.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.ands.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ands.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ands.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ands.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.bics.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.bics.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.bics.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.bics.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.eors.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.eors.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.eors.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.eors.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.orr.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.orr.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.orr.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.orr.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.orn.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.orn.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.orn.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.orn.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.nor.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.nor.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.nor.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.nor.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.nand.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.nand.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.nand.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.nand.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.orrs.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.orrs.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.orrs.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.orrs.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.orns.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.orns.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.orns.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.orns.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.nors.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.nors.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.nors.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.nors.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.nands.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.nands.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.nands.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.nands.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>)
