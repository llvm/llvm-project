; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; Test the default data layout import.

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-DAG:   "dlti.endianness" = "little"
; CHECK-DAG:   i1 = dense<8> : vector<2xi64>
; CHECK-DAG:   i8 = dense<8> : vector<2xi64>
; CHECK-DAG:   i16 = dense<16> : vector<2xi64>
; CHECK-DAG:   i32 = dense<32> : vector<2xi64>
; CHECK-DAG:   i64 = dense<[32, 64]> : vector<2xi64>
; CHECK-DAG:   !llvm.ptr = dense<64> : vector<4xi64>
; CHECK-DAG:   f16 = dense<16> : vector<2xi64>
; CHECK-DAG:   f64 = dense<64> : vector<2xi64>
; CHECK-DAG:   f128 = dense<128> : vector<2xi64>
; CHECK: >
target datalayout = ""

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-DAG:   "dlti.endianness" = "little"
; CHECK-DAG:   i64 = dense<64> : vector<2xi64>
; CHECK-DAG:   f80 = dense<128> : vector<2xi64>
; CHECK-DAG:   i8 = dense<8> : vector<2xi64>
; CHECK-DAG:   !llvm.ptr<270> = dense<[32, 64, 64, 32]> : vector<4xi64>
; CHECK-DAG:   !llvm.ptr<271> = dense<32> : vector<4xi64>
; CHECK-DAG:   !llvm.ptr<272> = dense<64> : vector<4xi64>
; CHECK-DAG:   "dlti.stack_alignment" = 128 : i64
target datalayout = "e-m:e-p270:32:64-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-DAG:   "dlti.endianness" = "big"
; CHECK-DAG:   !llvm.ptr<270> = dense<[16, 32, 64, 8]> : vector<4xi64>
; CHECK-DAG:   !llvm.ptr<271> = dense<[16, 32, 64, 16]> : vector<4xi64>
; CHECK-DAG:   "dlti.alloca_memory_space" = 1 : ui64
; CHECK-DAG:   i64 = dense<[64, 128]> : vector<2xi64>
target datalayout = "A1-E-p270:16:32:64:8-p271:16:32:64-i64:64:128"

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-NOT:   "dlti.alloca_memory_space" =
target datalayout = "A0"
