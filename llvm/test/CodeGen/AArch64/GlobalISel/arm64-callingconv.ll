; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

; CHECK-LABEL: name: args_i32
; CHECK: %[[ARG0:[0-9]+]](32) = COPY %w0
; CHECK: %{{[0-9]+}}(32) = COPY %w1
; CHECK: %{{[0-9]+}}(32) = COPY %w2
; CHECK: %{{[0-9]+}}(32) = COPY %w3
; CHECK: %{{[0-9]+}}(32) = COPY %w4
; CHECK: %{{[0-9]+}}(32) = COPY %w5
; CHECK: %{{[0-9]+}}(32) = COPY %w6
; CHECK: %{{[0-9]+}}(32) = COPY %w7
; CHECK: %w0 = COPY %[[ARG0]]

define i32 @args_i32(i32 %w0, i32 %w1, i32 %w2, i32 %w3,
                     i32 %w4, i32 %w5, i32 %w6, i32 %w7) {
  ret i32 %w0
}

; CHECK-LABEL: name: args_i64
; CHECK: %[[ARG0:[0-9]+]](64) = COPY %x0
; CHECK: %{{[0-9]+}}(64) = COPY %x1
; CHECK: %{{[0-9]+}}(64) = COPY %x2
; CHECK: %{{[0-9]+}}(64) = COPY %x3
; CHECK: %{{[0-9]+}}(64) = COPY %x4
; CHECK: %{{[0-9]+}}(64) = COPY %x5
; CHECK: %{{[0-9]+}}(64) = COPY %x6
; CHECK: %{{[0-9]+}}(64) = COPY %x7
; CHECK: %x0 = COPY %[[ARG0]]
define i64 @args_i64(i64 %x0, i64 %x1, i64 %x2, i64 %x3,
                     i64 %x4, i64 %x5, i64 %x6, i64 %x7) {
  ret i64 %x0
}


; CHECK-LABEL: name: args_ptrs
; CHECK: %[[ARG0:[0-9]+]](64) = COPY %x0
; CHECK: %{{[0-9]+}}(64) = COPY %x1
; CHECK: %{{[0-9]+}}(64) = COPY %x2
; CHECK: %{{[0-9]+}}(64) = COPY %x3
; CHECK: %{{[0-9]+}}(64) = COPY %x4
; CHECK: %{{[0-9]+}}(64) = COPY %x5
; CHECK: %{{[0-9]+}}(64) = COPY %x6
; CHECK: %{{[0-9]+}}(64) = COPY %x7
; CHECK: %x0 = COPY %[[ARG0]]
define i8* @args_ptrs(i8* %x0, i16* %x1, <2 x i8>* %x2, {i8, i16, i32}* %x3,
                      [3 x float]* %x4, double* %x5, i8* %x6, i8* %x7) {
  ret i8* %x0
}
