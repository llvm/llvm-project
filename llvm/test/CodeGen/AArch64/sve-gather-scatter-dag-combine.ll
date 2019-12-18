; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

; Verify that DAG combine rules for LD1 + sext/zext don't apply when the
; result of LD1 has multiple uses

define <vscale x 2 x i64> @no_dag_combine_zext_sext(<vscale x 2 x i1> %pg,
                                                    <vscale x 2 x i64> %base,
                                                    <vscale x 2 x i8>* %res_out,
                                                    <vscale x 2 x i1> %pred) {
; CHECK-LABEL: no_dag_combine_zext_sext
; CHECK:  	ld1b	{ z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT:	st1b	{ z0.d }, p1, [x0]
; CHECK-NEXT:	and	z0.d, z0.d, #0xff
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.imm.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                 <vscale x 2 x i64> %base,
                                                                                 i64 16)
  %res1 = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  %res2 = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %load,
                                      <vscale x 2 x i8> *%res_out,
                                      i32 8,
                                      <vscale x 2 x i1> %pred)

  ret <vscale x 2 x i64> %res1
}

define <vscale x 2 x i64> @no_dag_combine_sext(<vscale x 2 x i1> %pg,
                                               <vscale x 2 x i64> %base,
                                               <vscale x 2 x i8>* %res_out,
                                               <vscale x 2 x i1> %pred) {
; CHECK-LABEL: no_dag_combine_sext
; CHECK:  	ld1b	{ z1.d }, p0/z, [z0.d, #16]
; CHECK-NEXT:	ptrue	p0.d
; CHECK-NEXT:	sxtb	z0.d, p0/m, z1.d
; CHECK-NEXT:	st1b	{ z1.d }, p1, [x0]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.imm.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                 <vscale x 2 x i64> %base,
                                                                                 i64 16)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %load,
                                      <vscale x 2 x i8> *%res_out,
                                      i32 8,
                                      <vscale x 2 x i1> %pred)

  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @no_dag_combine_zext(<vscale x 2 x i1> %pg,
                                               <vscale x 2 x i64> %base,
                                               <vscale x 2 x i8>* %res_out,
                                               <vscale x 2 x i1> %pred) {
; CHECK-LABEL: no_dag_combine_zext
; CHECK:  	ld1b	{ z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT:	st1b	{ z0.d }, p1, [x0]
; CHECK-NEXT:	and	z0.d, z0.d, #0xff
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.imm.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                 <vscale x 2 x i64> %base,
                                                                                 i64 16)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %load,
                                      <vscale x 2 x i8> *%res_out,
                                      i32 8,
                                      <vscale x 2 x i1> %pred)

  ret <vscale x 2 x i64> %res
}

declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.imm.nxv2i8.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)
declare void @llvm.masked.store.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i8>*, i32, <vscale x 2 x i1>)
