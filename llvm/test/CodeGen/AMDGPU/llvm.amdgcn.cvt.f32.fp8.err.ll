; RUN: split-file %s %t

; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/fp8-byte0-err.ll 2>&1 | FileCheck -check-prefix=ERR-FP8-BYTE0-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/fp8-byte1-err.ll 2>&1 | FileCheck -check-prefix=ERR-FP8-BYTE1-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/bf8-byte0-err.ll 2>&1 | FileCheck -check-prefix=ERR-BF8-BYTE0-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/bf8-byte1-err.ll 2>&1 | FileCheck -check-prefix=ERR-BF8-BYTE1-ERR %s

; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/fp8-byte0-err.ll 2>&1 | FileCheck -check-prefix=ERR-FP8-BYTE0-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/fp8-byte1-err.ll 2>&1 | FileCheck -check-prefix=ERR-FP8-BYTE1-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/bf8-byte0-err.ll 2>&1 | FileCheck -check-prefix=ERR-BF8-BYTE0-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/bf8-byte1-err.ll 2>&1 | FileCheck -check-prefix=ERR-BF8-BYTE1-ERR %s


; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/fp8-byte0-err.ll 2>&1 | FileCheck -check-prefix=ERR-FP8-BYTE0-ERR-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/fp8-byte1-err.ll 2>&1 | FileCheck -check-prefix=ERR-FP8-BYTE1-ERR-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/bf8-byte0-err.ll 2>&1 | FileCheck -check-prefix=ERR-BF8-BYTE0-ERR-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/bf8-byte1-err.ll 2>&1 | FileCheck -check-prefix=ERR-BF8-BYTE1-ERR-GISEL %s

; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/fp8-byte0-err.ll 2>&1 | FileCheck -check-prefix=ERR-FP8-BYTE0-ERR-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/fp8-byte1-err.ll 2>&1 | FileCheck -check-prefix=ERR-FP8-BYTE1-ERR-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/bf8-byte0-err.ll 2>&1 | FileCheck -check-prefix=ERR-BF8-BYTE0-ERR-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/bf8-byte1-err.ll 2>&1 | FileCheck -check-prefix=ERR-BF8-BYTE1-ERR-GISEL %s



; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/pk-fp8-word0-err.ll 2>&1 | FileCheck -check-prefix=ERR-PK-FP8-WORD0-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/pk-fp8-word1-err.ll 2>&1 | FileCheck -check-prefix=ERR-PK-FP8-WORD1-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/pk-bf8-word0-err.ll 2>&1 | FileCheck -check-prefix=ERR-PK-BF8-WORD0-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx908 -filetype=null %t/pk-bf8-word1-err.ll 2>&1 | FileCheck -check-prefix=ERR-PK-BF8-WORD1-ERR %s

; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/pk-fp8-word0-err.ll 2>&1 | FileCheck -check-prefix=ERR-PK-FP8-WORD0-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/pk-fp8-word1-err.ll 2>&1 | FileCheck -check-prefix=ERR-PK-FP8-WORD1-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/pk-bf8-word0-err.ll 2>&1 | FileCheck -check-prefix=ERR-PK-BF8-WORD0-ERR %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %t/pk-bf8-word1-err.ll 2>&1 | FileCheck -check-prefix=ERR-PK-BF8-WORD1-ERR %s


;--- fp8-byte0-err.ll
; ERR-FP8-BYTE0-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.f32.fp8
; ERR-FP8-BYTE0-ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC intrinsic(@llvm.amdgcn.cvt.f32.fp8), %{{[0-9]+}}:vgpr(s32), 0

define float @test_cvt_f32_fp8_byte0(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.fp8(i32 %a, i32 0)
  ret float %ret
}

;--- fp8-byte1-err.ll
; ERR-FP8-BYTE1-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.f32.fp8
; ERR-FP8-BYTE1-ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC intrinsic(@llvm.amdgcn.cvt.f32.fp8), %{{[0-9]+}}:vgpr(s32), 1
define float @test_cvt_f32_fp8_byte1(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.fp8(i32 %a, i32 1)
  ret float %ret
}

;--- bf8-byte0-err.ll
; ERR-BF8-BYTE0-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.f32.bf8
; ERR-BF8-BYTE0-ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC intrinsic(@llvm.amdgcn.cvt.f32.bf8), %{{[0-9]+}}:vgpr(s32), 0
define float @test_cvt_f32_bf8_byte0(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.bf8(i32 %a, i32 0)
  ret float %ret
}

;--- bf8-byte1-err.ll
; ERR-BF8-BYTE1-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.f32.bf8
; ERR-BF8-BYTE1-ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC intrinsic(@llvm.amdgcn.cvt.f32.bf8), %{{[0-9]+}}:vgpr(s32), 1
define float @test_cvt_f32_bf8_byte1(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.bf8(i32 %a, i32 1)
  ret float %ret
}

;--- pk-fp8-word0-err.ll
; ERR-PK-FP8-WORD0-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.pk.f32.fp8
; ERR-PK-FP8-WORD0-ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC intrinsic(@llvm.amdgcn.cvt.pk.f32.fp8), %{{[0-9]+}}:vgpr(s32), 0
define <2 x float> @test_cvt_pk_f32_fp8_word0(i32 %a) {
  %ret = tail call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %a, i1 false)
  ret <2 x float> %ret
}

;--- pk-fp8-word1-err.ll
; ERR-PK-FP8-WORD1-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.pk.f32.fp8
; ERR-PK-FP8-WORD1-ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC intrinsic(@llvm.amdgcn.cvt.pk.f32.fp8), %{{[0-9]+}}:vgpr(s32), 1
define <2 x float> @test_cvt_pk_f32_fp8_word1(i32 %a) {
  %ret = tail call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %a, i1 true)
  ret <2 x float> %ret
}

;--- pk-bf8-word0-err.ll
; ERR-PK-BF8-WORD0-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.pk.f32.bf8
; ERR-PK-BF8-WORD0-ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC intrinsic(@llvm.amdgcn.cvt.pk.f32.bf8), %{{[0-9]+}}:vgpr(s32), 0
define <2 x float> @test_cvt_pk_f32_bf8_word0(i32 %a) {
  %ret = tail call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %a, i1 false)
  ret <2 x float> %ret
}

;--- pk-bf8-word1-err.ll
; ERR-PK-BF8-WORD1-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.pk.f32.bf8
; ERR-PK-BF8-WORD1-ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC intrinsic(@llvm.amdgcn.cvt.pk.f32.bf8), %{{[0-9]+}}:vgpr(s32), 1
define <2 x float> @test_cvt_pk_f32_bf8_word1(i32 %a) {
  %ret = tail call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %a, i1 true)
  ret <2 x float> %ret
}
