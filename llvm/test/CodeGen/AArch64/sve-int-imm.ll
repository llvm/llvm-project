; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

define <vscale x 16 x i8> @add_imm_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: add_imm_i8_low
; CHECK: add  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 16 x i8> @llvm.aarch64.sve.add.imm.nxv16i8(<vscale x 16 x i8> %a,
                                                                    i32 30)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @add_imm_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: add_imm_i16_low
; CHECK: add  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.add.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                    i32 30)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @add_imm_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: add_imm_i16_high
; CHECK: add  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.add.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                    i32 1024)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @add_imm_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: add_imm_i32_low
; CHECK: add  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.add.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                    i32 30)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @add_imm_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: add_imm_i32_high
; CHECK: add  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.add.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                    i32 1024)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @add_imm_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: add_imm_i64_low
; CHECK: add  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.add.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                    i32 30)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @add_imm_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: add_imm_i64_high
; CHECK: add  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.add.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                    i32 1024)
  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @sub_imm_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sub_imm_i8_low
; CHECK: sub  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 16 x i8> @llvm.aarch64.sve.sub.imm.nxv16i8(<vscale x 16 x i8> %a,
                                                                    i32 30)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @sub_imm_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sub_imm_i16_low
; CHECK: sub  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.sub.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                    i32 30)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @sub_imm_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sub_imm_i16_high
; CHECK: sub  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.sub.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                    i32 1024)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sub_imm_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sub_imm_i32_low
; CHECK: sub  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.sub.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                    i32 30)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sub_imm_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sub_imm_i32_high
; CHECK: sub  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.sub.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                    i32 1024)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sub_imm_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sub_imm_i64_low
; CHECK: sub  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.sub.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                    i32 30)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sub_imm_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sub_imm_i64_high
; CHECK: sub  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.sub.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                    i32 1024)
  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @subr_imm_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: subr_imm_i8_low
; CHECK: subr  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 16 x i8> @llvm.aarch64.sve.subr.imm.nxv16i8(<vscale x 16 x i8> %a,
                                                                     i32 30)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @subr_imm_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: subr_imm_i16_low
; CHECK: subr  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.subr.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                     i32 30)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @subr_imm_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: subr_imm_i16_high
; CHECK: subr  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.subr.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                     i32 1024)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @subr_imm_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: subr_imm_i32_low
; CHECK: subr  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.subr.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                     i32 30)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @subr_imm_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: subr_imm_i32_high
; CHECK: subr  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.subr.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                     i32 1024)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @subr_imm_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: subr_imm_i64_low
; CHECK: subr  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.subr.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                     i32 30)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @subr_imm_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: subr_imm_i64_high
; CHECK: subr  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.subr.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                     i32 1024)
  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @sqadd_imm_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sqadd_imm_i8_low
; CHECK: sqadd  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 16 x i8> @llvm.aarch64.sve.sqadd.imm.nxv16i8(<vscale x 16 x i8> %a,
                                                                      i32 30)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @sqadd_imm_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqadd_imm_i16_low
; CHECK: sqadd  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.sqadd.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                      i32 30)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @sqadd_imm_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqadd_imm_i16_high
; CHECK: sqadd  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.sqadd.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                      i32 1024)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqadd_imm_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqadd_imm_i32_low
; CHECK: sqadd  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.sqadd.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                      i32 30)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sqadd_imm_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqadd_imm_i32_high
; CHECK: sqadd  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.sqadd.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                      i32 1024)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqadd_imm_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqadd_imm_i64_low
; CHECK: sqadd  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.sqadd.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                      i32 30)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sqadd_imm_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqadd_imm_i64_high
; CHECK: sqadd  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.sqadd.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                      i32 1024)
  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @uqadd_imm_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uqadd_imm_i8_low
; CHECK: uqadd  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 16 x i8> @llvm.aarch64.sve.uqadd.imm.nxv16i8(<vscale x 16 x i8> %a,
                                                                      i32 30)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @uqadd_imm_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqadd_imm_i16_low
; CHECK: uqadd  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.uqadd.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                      i32 30)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @uqadd_imm_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqadd_imm_i16_high
; CHECK: uqadd  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.uqadd.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                      i32 1024)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @uqadd_imm_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqadd_imm_i32_low
; CHECK: uqadd  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.uqadd.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                      i32 30)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @uqadd_imm_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqadd_imm_i32_high
; CHECK: uqadd  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.uqadd.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                      i32 1024)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @uqadd_imm_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqadd_imm_i64_low
; CHECK: uqadd  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.uqadd.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                      i32 30)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @uqadd_imm_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqadd_imm_i64_high
; CHECK: uqadd  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.uqadd.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                      i32 1024)
  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @sqsub_imm_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sqsub_imm_i8_low
; CHECK: sqsub  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 16 x i8> @llvm.aarch64.sve.sqsub.imm.nxv16i8(<vscale x 16 x i8> %a,
                                                                      i32 30)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @sqsub_imm_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqsub_imm_i16_low
; CHECK: sqsub  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.sqsub.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                      i32 30)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @sqsub_imm_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqsub_imm_i16_high
; CHECK: sqsub  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.sqsub.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                      i32 1024)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqsub_imm_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqsub_imm_i32_low
; CHECK: sqsub  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.sqsub.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                      i32 30)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sqsub_imm_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqsub_imm_i32_high
; CHECK: sqsub  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.sqsub.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                      i32 1024)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqsub_imm_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqsub_imm_i64_low
; CHECK: sqsub  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.sqsub.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                      i32 30)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sqsub_imm_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqsub_imm_i64_high
; CHECK: sqsub  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.sqsub.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                      i32 1024)
  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @uqsub_imm_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uqsub_imm_i8_low
; CHECK: uqsub  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 16 x i8> @llvm.aarch64.sve.uqsub.imm.nxv16i8(<vscale x 16 x i8> %a,
                                                                      i32 30)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @uqsub_imm_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqsub_imm_i16_low
; CHECK: uqsub  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.uqsub.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                      i32 30)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @uqsub_imm_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqsub_imm_i16_high
; CHECK: uqsub  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.uqsub.imm.nxv8i16(<vscale x 8 x i16> %a,
                                                                      i32 1024)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @uqsub_imm_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqsub_imm_i32_low
; CHECK: uqsub  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.uqsub.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                      i32 30)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @uqsub_imm_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqsub_imm_i32_high
; CHECK: uqsub  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.uqsub.imm.nxv4i32(<vscale x 4 x i32> %a,
                                                                      i32 1024)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @uqsub_imm_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqsub_imm_i64_low
; CHECK: uqsub  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.uqsub.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                      i32 30)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @uqsub_imm_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqsub_imm_i64_high
; CHECK: uqsub  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.uqsub.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                      i32 1024)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.add.imm.nxv16i8(<vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.add.imm.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.add.imm.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.add.imm.nxv2i64(<vscale x 2 x i64>, i32)
declare <vscale x 16 x i8> @llvm.aarch64.sve.sub.imm.nxv16i8(<vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sub.imm.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sub.imm.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sub.imm.nxv2i64(<vscale x 2 x i64>, i32)
declare <vscale x 16 x i8> @llvm.aarch64.sve.subr.imm.nxv16i8(<vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.subr.imm.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.subr.imm.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.subr.imm.nxv2i64(<vscale x 2 x i64>, i32)
declare <vscale x 16 x i8> @llvm.aarch64.sve.sqadd.imm.nxv16i8(<vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqadd.imm.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqadd.imm.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqadd.imm.nxv2i64(<vscale x 2 x i64>, i32)
declare <vscale x 16 x i8> @llvm.aarch64.sve.uqadd.imm.nxv16i8(<vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqadd.imm.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqadd.imm.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uqadd.imm.nxv2i64(<vscale x 2 x i64>, i32)
declare <vscale x 16 x i8> @llvm.aarch64.sve.sqsub.imm.nxv16i8(<vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqsub.imm.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqsub.imm.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqsub.imm.nxv2i64(<vscale x 2 x i64>, i32)
declare <vscale x 16 x i8> @llvm.aarch64.sve.uqsub.imm.nxv16i8(<vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqsub.imm.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqsub.imm.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uqsub.imm.nxv2i64(<vscale x 2 x i64>, i32)
