; RUN: llc -mtriple=x86_64-linux-gnu -global-isel -stop-after=irtranslator < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=X64

define <4 x i32> @test_v4i32_args(<4 x i32> %arg1, <4 x i32> %arg2) {
; X64: name:            test_v4i32_args
; X64: liveins: %xmm0, %xmm1
; X64:      [[ARG1:%[0-9]+]](<4 x s32>) = COPY %xmm0
; X64-NEXT: [[ARG2:%[0-9]+]](<4 x s32>) = COPY %xmm1
; X64-NEXT: %xmm0 = COPY [[ARG2:%[0-9]+]](<4 x s32>)
; X64-NEXT: RET 0, implicit %xmm0
  ret <4 x i32> %arg2
}

define <8 x i32> @test_v8i32_args(<8 x i32> %arg1) {
; X64: name:            test_v8i32_args
; X64: liveins: %xmm0, %xmm1
; X64:      [[ARG1L:%[0-9]+]](<4 x s32>) = COPY %xmm0
; X64-NEXT: [[ARG1H:%[0-9]+]](<4 x s32>) = COPY %xmm1
; X64-NEXT: [[ARG1:%[0-9]+]](<8 x s32>) = G_SEQUENCE [[ARG1L:%[0-9]+]](<4 x s32>), 0, [[ARG1H:%[0-9]+]](<4 x s32>), 128
; X64-NEXT: [[RETL:%[0-9]+]](<4 x s32>), [[RETH:%[0-9]+]](<4 x s32>) = G_EXTRACT [[ARG1:%[0-9]+]](<8 x s32>), 0, 128
; X64-NEXT: %xmm0 = COPY [[RETL:%[0-9]+]](<4 x s32>)
; X64-NEXT: %xmm1 = COPY [[RETH:%[0-9]+]](<4 x s32>)
; X64-NEXT: RET 0, implicit %xmm0, implicit %xmm1

  ret <8 x i32> %arg1
}
