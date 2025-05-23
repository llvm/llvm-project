; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; Test the default data layout import.

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-SAME:   !llvm.ptr = dense<64> : vector<4xi64>
; CHECK-SAME:   i1 = dense<8> : vector<2xi64>
; CHECK-SAME:   i8 = dense<8> : vector<2xi64>
; CHECK-SAME:   i16 = dense<16> : vector<2xi64>
; CHECK-SAME:   i32 = dense<32> : vector<2xi64>
; CHECK-SAME:   i64 = dense<[32, 64]> : vector<2xi64>
; CHECK-SAME:   f16 = dense<16> : vector<2xi64>
; CHECK-SAME:   f64 = dense<64> : vector<2xi64>
; CHECK-SAME:   f128 = dense<128> : vector<2xi64>
; CHECK-SAME:   "dlti.endianness" = "little"
; CHECK: >
target datalayout = ""

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-SAME:   !llvm.ptr<270> = dense<[32, 64, 64, 32]> : vector<4xi64>
; CHECK-SAME:   !llvm.ptr<271> = dense<32> : vector<4xi64>
; CHECK-SAME:   !llvm.ptr<272> = dense<64> : vector<4xi64>
; CHECK-SAME:   i64 = dense<64> : vector<2xi64>
; CHECK-SAME:   f80 = dense<128> : vector<2xi64>
; CHECK-SAME:   i8 = dense<8> : vector<2xi64>
; CHECK-SAME:   "dlti.endianness" = "little"
; CHECK-SAME:   "dlti.mangling_mode" = "e"
; CHECK-SAME:   "dlti.stack_alignment" = 128 : i64
; CHECK-SAME:   "dlti.function_pointer_alignment" = #dlti.function_pointer_alignment<32, function_dependent = true>
target datalayout = "e-m:e-p270:32:64-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-Fn32"

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-SAME:   !llvm.ptr<270> = dense<[16, 32, 64, 8]> : vector<4xi64>
; CHECK-SAME:   !llvm.ptr<271> = dense<[16, 32, 64, 16]> : vector<4xi64>
; CHECK-SAME:   i64 = dense<[64, 128]> : vector<2xi64>
; CHECK-SAME:   "dlti.alloca_memory_space" = 1 : ui64
; CHECK-SAME:   "dlti.endianness" = "big"
target datalayout = "A1-E-p270:16:32:64:8-p271:16:32:64-i64:64:128"

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-NOT:   "dlti.alloca_memory_space" =
target datalayout = "A0"
