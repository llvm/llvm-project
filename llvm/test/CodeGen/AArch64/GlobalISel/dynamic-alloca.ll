; RUN: llc -mtriple=aarch64 -global-isel %s -o - -stop-after=irtranslator | FileCheck %s

; CHECK-LABEL: name: test_simple_alloca
; CHECK: [[NUMELTS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[NUMELTS_64:%[0-9]+]](s64) = G_ZEXT [[NUMELTS]](s32)
; CHECK: [[TYPE_SIZE:%[0-9]+]](s64) = G_CONSTANT i64 1
; CHECK: [[NUMBYTES:%[0-9]+]](s64) = G_MUL [[NUMELTS_64]], [[TYPE_SIZE]]
; CHECK: [[SP_TMP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[SP_INT:%[0-9]+]](s64) = G_PTRTOINT [[SP_TMP]](p0)
; CHECK: [[ALLOC:%[0-9]+]](s64) = G_SUB [[SP_INT]], [[NUMBYTES]]
; CHECK: [[ALIGN_M_1:%[0-9]+]](s64) = G_CONSTANT i64 15
; CHECK: [[ALIGN_TMP:%[0-9]+]](s64) = G_SUB [[ALLOC]], [[ALIGN_M_1]]
; CHECK: [[ALIGN_MASK:%[0-9]+]](s64) = G_CONSTANT i64 -16
; CHECK: [[ALIGNED_ALLOC:%[0-9]+]](s64) = G_AND [[ALIGN_TMP]], [[ALIGN_MASK]]
; CHECK: [[ALLOC_PTR:%[0-9]+]](p0) = G_INTTOPTR [[ALIGNED_ALLOC]](s64)
; CHECK: %sp = COPY [[ALLOC_PTR]]
; CHECK: %x0 = COPY [[ALLOC_PTR]]
define i8* @test_simple_alloca(i32 %numelts) {
  %addr = alloca i8, i32 %numelts
  ret i8* %addr
}

; CHECK-LABEL: name: test_aligned_alloca
; CHECK: [[NUMELTS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[NUMELTS_64:%[0-9]+]](s64) = G_ZEXT [[NUMELTS]](s32)
; CHECK: [[TYPE_SIZE:%[0-9]+]](s64) = G_CONSTANT i64 1
; CHECK: [[NUMBYTES:%[0-9]+]](s64) = G_MUL [[NUMELTS_64]], [[TYPE_SIZE]]
; CHECK: [[SP_TMP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[SP_INT:%[0-9]+]](s64) = G_PTRTOINT [[SP_TMP]](p0)
; CHECK: [[ALLOC:%[0-9]+]](s64) = G_SUB [[SP_INT]], [[NUMBYTES]]
; CHECK: [[ALIGN_M_1:%[0-9]+]](s64) = G_CONSTANT i64 31
; CHECK: [[ALIGN_TMP:%[0-9]+]](s64) = G_SUB [[ALLOC]], [[ALIGN_M_1]]
; CHECK: [[ALIGN_MASK:%[0-9]+]](s64) = G_CONSTANT i64 -32
; CHECK: [[ALIGNED_ALLOC:%[0-9]+]](s64) = G_AND [[ALIGN_TMP]], [[ALIGN_MASK]]
; CHECK: [[ALLOC_PTR:%[0-9]+]](p0) = G_INTTOPTR [[ALIGNED_ALLOC]](s64)
; CHECK: %sp = COPY [[ALLOC_PTR]]
; CHECK: %x0 = COPY [[ALLOC_PTR]]
define i8* @test_aligned_alloca(i32 %numelts) {
  %addr = alloca i8, i32 %numelts, align 32
  ret i8* %addr
}

; CHECK-LABEL: name: test_natural_alloca
; CHECK: [[NUMELTS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[NUMELTS_64:%[0-9]+]](s64) = G_ZEXT [[NUMELTS]](s32)
; CHECK: [[TYPE_SIZE:%[0-9]+]](s64) = G_CONSTANT i64 16
; CHECK: [[NUMBYTES:%[0-9]+]](s64) = G_MUL [[NUMELTS_64]], [[TYPE_SIZE]]
; CHECK: [[SP_TMP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[SP_INT:%[0-9]+]](s64) = G_PTRTOINT [[SP_TMP]](p0)
; CHECK: [[ALLOC:%[0-9]+]](s64) = G_SUB [[SP_INT]], [[NUMBYTES]]
; CHECK: [[ALLOC_PTR:%[0-9]+]](p0) = G_INTTOPTR [[ALLOC]](s64)
; CHECK: %sp = COPY [[ALLOC_PTR]]
; CHECK: %x0 = COPY [[ALLOC_PTR]]
define i128* @test_natural_alloca(i32 %numelts) {
  %addr = alloca i128, i32 %numelts
  ret i128* %addr
}
