; RUN: llc -mtriple=aarch64-linux-gnu -stop-after=irtranslator -enable-cse-in-irtranslator=1 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; CHECK-LABEL: name: test_split_struct
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[LO:%[0-9]+]]:_(s64) = G_LOAD %0(p0) :: (load (s64) from %ir.ptr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST]](s64)
; CHECK: [[HI:%[0-9]+]]:_(s64) = G_LOAD [[GEP]](p0) :: (load (s64) from %ir.ptr + 8)

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[CST2]](s64)
; CHECK: G_STORE [[LO]](s64), [[GEP2]](p0) :: (store (s64) into stack, align 1)
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[CST]](s64)
; CHECK: G_STORE [[HI]](s64), [[GEP3]](p0) :: (store (s64) into stack + 8, align 1)
define void @test_split_struct([2 x i64]* %ptr) {
  %struct = load [2 x i64], [2 x i64]* %ptr
  call void @take_split_struct([2 x i64]* null, i64 1, i64 2, i64 3,
                               i64 4, i64 5, i64 6,
                               [2 x i64] %struct)
  ret void
}

declare void @take_split_struct([2 x i64]* %ptr, i64, i64, i64,
                               i64, i64, i64,
                               [2 x i64] %in) ;
