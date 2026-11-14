; RUN: split-file %s %t
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=null %t/fadd.ll
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -filetype=null %t/fadd-streaming.ll
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -filetype=null %t/sme-zero-compatible.ll
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme2 -filetype=null %t/sme-zero-zt-both-modes.ll
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=null %t/st2-pointers.ll
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=null %t/fadd-compatible.ll
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=null %t/add-supported.ll
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=null %t/add-unsupported.ll 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-TYPES
; RUN: not llc -mtriple=aarch64-linux-gnu -filetype=null %t/sme-zero-compatible.ll 2>&1 | FileCheck %s --check-prefix=SME-FEATURE
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sme -filetype=null %t/sme-zero-zt-both-modes.ll 2>&1 | FileCheck %s --check-prefix=ZT-NEEDS-SME2
; RUN: not llc -mtriple=aarch64-linux-gnu -filetype=null %t/fadd.ll 2>&1 | FileCheck %s --check-prefix=NO-SVE
; RUN: not llc -mtriple=aarch64-linux-gnu -global-isel -global-isel-abort=0 -filetype=null %t/fadd.ll 2>&1 | FileCheck %s --check-prefix=NO-SVE
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=null %t/fadd-streaming.ll 2>&1 | FileCheck %s --check-prefix=STREAMING-SME
; RUN: not llc -mtriple=aarch64-linux-gnu -filetype=null %t/st2-pointers.ll 2>&1 | FileCheck %s --check-prefix=POINTER-VECTOR
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sme -filetype=null %t/ldnf-streaming.ll 2>&1 | FileCheck %s --check-prefix=BAD-MODE
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=null %t/fadd-bf16.ll 2>&1 | FileCheck %s --check-prefix=BF16
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sve,+sve-b16b16 -force-streaming-compatible -filetype=null %t/fadd-bf16.ll 2>&1 | FileCheck %s --check-prefix=COMPATIBLE-EXT
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sme -filetype=null %t/sme-load-nonstreaming.ll 2>&1 | FileCheck %s --check-prefix=SME-MODE
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sme2,+sve-b16b16 -filetype=null %t/bfclamp-nonstreaming.ll 2>&1 | FileCheck %s --check-prefix=BFCLAMP-MODE
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sme2p1 -filetype=null %t/fmaxnmqv-nonstreaming.ll 2>&1 | FileCheck %s --check-prefix=SME2P1-NO-SVE

; NO-SVE: AArch64 intrinsic 'llvm.aarch64.sve.fadd.nxv4f32' in function 'fadd' requires sve in non-streaming mode
; STREAMING-SME: streaming SVE functions require SME
; POINTER-VECTOR: AArch64 intrinsic 'llvm.aarch64.sve.st2.nxv2p0.p0' in function 'st2_pointers' requires sve in non-streaming mode
; BAD-MODE: AArch64 intrinsic 'llvm.aarch64.sve.ldnf1.nxv4f32.p0' in function 'ldnf_streaming' is unavailable in streaming mode
; BF16: AArch64 intrinsic 'llvm.aarch64.sve.fadd.nxv8bf16' in function 'fadd_bf16' requires sve,sve-b16b16 in non-streaming mode
; COMPATIBLE-EXT: AArch64 intrinsic 'llvm.aarch64.sve.fadd.nxv8bf16' in function 'fadd_bf16' requires sme2,sve-b16b16 in streaming mode
; SME-FEATURE: AArch64 intrinsic 'llvm.aarch64.sme.zero' in function 'sme_zero_nonstreaming' requires sme in non-streaming mode
; ZT-NEEDS-SME2: AArch64 intrinsic 'llvm.aarch64.sme.zero.zt' in function 'zero_zt_nonstreaming' requires sme2 in non-streaming mode
; SME-MODE: AArch64 intrinsic 'llvm.aarch64.sme.ld1b.horiz.p0' in function 'sme_load_nonstreaming' is unavailable in non-streaming mode
; BFCLAMP-MODE: AArch64 intrinsic 'llvm.aarch64.sve.bfclamp.single.x2.nxv8bf16' in function 'bfclamp_nonstreaming' is unavailable in non-streaming mode
; SME2P1-NO-SVE: AArch64 intrinsic 'llvm.aarch64.sve.fmaxnmqv.v4f32.nxv4f32' in function 'fmaxnmqv_nonstreaming' requires sve,(sve2p1|sme2p1) in non-streaming mode
; UNSUPPORTED-TYPES: AArch64 intrinsic 'llvm.aarch64.sve.add.nxv16i16' in function 'add_unsupported' has an unsupported type signature

;--- add-supported.ll
declare <vscale x 8 x i16> @llvm.aarch64.sve.add.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)

define <vscale x 8 x i16> @add_supported(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
  %result = call <vscale x 8 x i16> @llvm.aarch64.sve.add.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %result
}

;--- add-unsupported.ll
declare <vscale x 16 x i16> @llvm.aarch64.sve.add.nxv16i16(<vscale x 16 x i1>, <vscale x 16 x i16>, <vscale x 16 x i16>)

define <vscale x 16 x i16> @add_unsupported(<vscale x 16 x i1> %pg, <vscale x 16 x i16> %a, <vscale x 16 x i16> %b) {
  %result = call <vscale x 16 x i16> @llvm.aarch64.sve.add.nxv16i16(<vscale x 16 x i1> %pg, <vscale x 16 x i16> %a, <vscale x 16 x i16> %b)
  ret <vscale x 16 x i16> %result
}

;--- fadd.ll
declare <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)

define <vscale x 4 x float> @fadd(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
  %result = call <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %result
}

;--- fadd-streaming.ll
declare <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)

define <vscale x 4 x float> @fadd_streaming(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) "aarch64_pstate_sm_enabled" {
  %result = call <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %result
}

;--- ldnf-streaming.ll
declare <vscale x 4 x float> @llvm.aarch64.sve.ldnf1.nxv4f32(<vscale x 4 x i1>, ptr)

define <vscale x 4 x float> @ldnf_streaming(<vscale x 4 x i1> %pg, ptr %p) "aarch64_pstate_sm_enabled" {
  %result = call <vscale x 4 x float> @llvm.aarch64.sve.ldnf1.nxv4f32(<vscale x 4 x i1> %pg, ptr %p)
  ret <vscale x 4 x float> %result
}

;--- fadd-bf16.ll
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.fadd.nxv8bf16(<vscale x 8 x i1>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)

define <vscale x 8 x bfloat> @fadd_bf16(<vscale x 8 x i1> %pg, <vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b) {
  %result = call <vscale x 8 x bfloat> @llvm.aarch64.sve.fadd.nxv8bf16(<vscale x 8 x i1> %pg, <vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b)
  ret <vscale x 8 x bfloat> %result
}

;--- fadd-compatible.ll
declare <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)

define <vscale x 4 x float> @fadd_compatible(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) "aarch64_pstate_sm_compatible" {
  %result = call <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %result
}

;--- sme-zero-compatible.ll
declare void @llvm.aarch64.sme.zero(i32 immarg)

define void @sme_zero_nonstreaming() {
  call void @llvm.aarch64.sme.zero(i32 0)
  ret void
}

;--- sme-zero-zt-both-modes.ll
declare void @llvm.aarch64.sme.zero.zt(i32 immarg)

define void @zero_zt_nonstreaming() {
  call void @llvm.aarch64.sme.zero.zt(i32 0)
  ret void
}

define void @zero_zt_streaming() "aarch64_pstate_sm_enabled" {
  call void @llvm.aarch64.sme.zero.zt(i32 0)
  ret void
}

;--- st2-pointers.ll
declare void @llvm.aarch64.sve.st2.nxv2p0(<vscale x 2 x ptr>, <vscale x 2 x ptr>, <vscale x 2 x i1>, ptr)

define void @st2_pointers(<vscale x 2 x ptr> %a, <vscale x 2 x ptr> %b, <vscale x 2 x i1> %pg, ptr %base) {
  call void @llvm.aarch64.sve.st2.nxv2p0(<vscale x 2 x ptr> %a, <vscale x 2 x ptr> %b, <vscale x 2 x i1> %pg, ptr %base)
  ret void
}

;--- sme-load-nonstreaming.ll
declare void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1>, ptr, i32, i32)

define void @sme_load_nonstreaming(<vscale x 16 x i1> %pg, ptr %p) {
  call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %p, i32 0, i32 0)
  ret void
}

;--- bfclamp-nonstreaming.ll
declare { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.bfclamp.single.x2.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)

define { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @bfclamp_nonstreaming(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x bfloat> %d) {
  %result = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.bfclamp.single.x2.nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x bfloat> %d)
  ret { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } %result
}

;--- fmaxnmqv-nonstreaming.ll
declare <4 x float> @llvm.aarch64.sve.fmaxnmqv.v4f32(<vscale x 4 x i1>, <vscale x 4 x float>)

define <4 x float> @fmaxnmqv_nonstreaming(<vscale x 4 x i1> %pg, <vscale x 4 x float> %v) {
  %result = call <4 x float> @llvm.aarch64.sve.fmaxnmqv.v4f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %v)
  ret <4 x float> %result
}
