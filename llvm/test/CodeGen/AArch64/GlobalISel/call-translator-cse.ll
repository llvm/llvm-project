; RUN: llc -mtriple=aarch64-linux-gnu -O1 -stop-after=irtranslator -enable-cse-in-irtranslator=1 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; CHECK-LABEL: name: test_split_struct
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[LO:%[0-9]+]]:_(s64) = G_LOAD %0(p0) :: (load 8 from %ir.ptr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_GEP [[ADDR]], [[CST]](s64)
; CHECK: [[HI:%[0-9]+]]:_(s64) = G_LOAD [[GEP]](p0) :: (load 8 from %ir.ptr + 8)

; CHECK: [[IMPDEF:%[0-9]+]]:_(s128) = G_IMPLICIT_DEF
; CHECK: [[INS1:%[0-9]+]]:_(s128) = G_INSERT [[IMPDEF]], [[LO]](s64), 0
; CHECK: [[INS2:%[0-9]+]]:_(s128) = G_INSERT [[INS1]], [[HI]](s64), 64
; CHECK: [[EXTLO:%[0-9]+]]:_(s64) = G_EXTRACT [[INS2]](s128), 0
; CHECK: [[EXTHI:%[0-9]+]]:_(s64) = G_EXTRACT [[INS2]](s128), 64

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[CST2]](s64)
; CHECK: G_STORE [[EXTLO]](s64), [[GEP2]](p0) :: (store 8 into stack, align 0)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[CST3:%[0-9]+]]:_(s64) = COPY [[CST]]
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[CST3]](s64)
; CHECK: G_STORE [[EXTHI]](s64), [[GEP3]](p0) :: (store 8 into stack + 8, align 0)
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
