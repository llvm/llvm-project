; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK: dlti.dl_spec = 
; CHECK: #dlti.dl_spec<
; CHECK:   #dlti.dl_entry<"dlti.endianness", "little">
; CHECK:   #dlti.dl_entry<i64, dense<64> : vector<2xi32>>
; CHECK:   #dlti.dl_entry<f80, dense<128> : vector<2xi32>>
; CHECK:   #dlti.dl_entry<i8, dense<8> : vector<2xi32>>
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @foo()
