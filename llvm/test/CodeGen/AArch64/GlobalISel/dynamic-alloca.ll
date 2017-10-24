; RUN: llc -mtriple=aarch64 -global-isel %s -o - -stop-after=irtranslator | FileCheck %s

; CHECK-LABEL: name: test_simple_alloca
; CHECK: [[NUMELTS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[TYPE_SIZE:%[0-9]+]]:_(s64) = G_CONSTANT i64 -1
; CHECK: [[NUMELTS_64:%[0-9]+]]:_(s64) = G_ZEXT [[NUMELTS]](s32)
; CHECK: [[NUMBYTES:%[0-9]+]]:_(s64) = G_MUL [[NUMELTS_64]], [[TYPE_SIZE]]
; CHECK: [[SP_TMP:%[0-9]+]]:_(p0) = COPY %sp
; CHECK: [[ALLOC:%[0-9]+]]:_(p0) = G_GEP [[SP_TMP]], [[NUMBYTES]]
; CHECK: [[ALIGNED_ALLOC:%[0-9]+]]:_(p0) = G_PTR_MASK [[ALLOC]], 4
; CHECK: %sp = COPY [[ALIGNED_ALLOC]]
; CHECK: [[ALLOC:%[0-9]+]]:_(p0) = COPY [[ALIGNED_ALLOC]]
; CHECK: %x0 = COPY [[ALLOC]]
define i8* @test_simple_alloca(i32 %numelts) {
  %addr = alloca i8, i32 %numelts
  ret i8* %addr
}

; CHECK-LABEL: name: test_aligned_alloca
; CHECK: [[NUMELTS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[TYPE_SIZE:%[0-9]+]]:_(s64) = G_CONSTANT i64 -1
; CHECK: [[NUMELTS_64:%[0-9]+]]:_(s64) = G_ZEXT [[NUMELTS]](s32)
; CHECK: [[NUMBYTES:%[0-9]+]]:_(s64) = G_MUL [[NUMELTS_64]], [[TYPE_SIZE]]
; CHECK: [[SP_TMP:%[0-9]+]]:_(p0) = COPY %sp
; CHECK: [[ALLOC:%[0-9]+]]:_(p0) = G_GEP [[SP_TMP]], [[NUMBYTES]]
; CHECK: [[ALIGNED_ALLOC:%[0-9]+]]:_(p0) = G_PTR_MASK [[ALLOC]], 5
; CHECK: %sp = COPY [[ALIGNED_ALLOC]]
; CHECK: [[ALLOC:%[0-9]+]]:_(p0) = COPY [[ALIGNED_ALLOC]]
; CHECK: %x0 = COPY [[ALLOC]]
define i8* @test_aligned_alloca(i32 %numelts) {
  %addr = alloca i8, i32 %numelts, align 32
  ret i8* %addr
}

; CHECK-LABEL: name: test_natural_alloca
; CHECK: [[NUMELTS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[TYPE_SIZE:%[0-9]+]]:_(s64) = G_CONSTANT i64 -16
; CHECK: [[NUMELTS_64:%[0-9]+]]:_(s64) = G_ZEXT [[NUMELTS]](s32)
; CHECK: [[NUMBYTES:%[0-9]+]]:_(s64) = G_MUL [[NUMELTS_64]], [[TYPE_SIZE]]
; CHECK: [[SP_TMP:%[0-9]+]]:_(p0) = COPY %sp
; CHECK: [[ALLOC:%[0-9]+]]:_(p0) = G_GEP [[SP_TMP]], [[NUMBYTES]]
; CHECK: %sp = COPY [[ALLOC]]
; CHECK: [[ALLOC_TMP:%[0-9]+]]:_(p0) = COPY [[ALLOC]]
; CHECK: %x0 = COPY [[ALLOC_TMP]]
define i128* @test_natural_alloca(i32 %numelts) {
  %addr = alloca i128, i32 %numelts
  ret i128* %addr
}
