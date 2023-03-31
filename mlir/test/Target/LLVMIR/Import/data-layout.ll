; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; Test the default data layout import.

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-DAG:   #dlti.dl_entry<"dlti.endianness", "little">
; CHECK-DAG:   #dlti.dl_entry<i1, dense<8> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<i8, dense<8> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<i16, dense<16> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<i32, dense<32> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>
; CHECK-DAG:   #dlti.dl_entry<f16, dense<16> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<f64, dense<64> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<f128, dense<128> : vector<2xi32>>
; CHECK: >
target datalayout = ""

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-DAG:   #dlti.dl_entry<"dlti.endianness", "little">
; CHECK-DAG:   #dlti.dl_entry<i64, dense<64> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<f80, dense<128> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<i8, dense<8> : vector<2xi32>>
; CHECK-DAG:   #dlti.dl_entry<!llvm.ptr<270>, dense<[32, 64, 64, 32]> : vector<4xi32>>
; CHECK-DAG:   #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>
; CHECK-DAG:   #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>
target datalayout = "e-m:e-p270:32:64-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-DAG:   #dlti.dl_entry<"dlti.endianness", "big">
; CHECK-DAG:   #dlti.dl_entry<!llvm.ptr<270>, dense<[16, 32, 64, 128]> : vector<4xi32>>
; CHECK-DAG:   #dlti.dl_entry<"dlti.alloca_memory_space", 1 : ui32>
; CHECK-DAG:   #dlti.dl_entry<i64, dense<[64, 128]> : vector<2xi32>>
target datalayout = "E-p270:16:32:64:128-A1-i64:64:128"

; // -----

; CHECK: dlti.dl_spec =
; CHECK: #dlti.dl_spec<
; CHECK-NOT:   #dlti.dl_entry<"dlti.alloca_memory_space"
target datalayout = "E-A0"
