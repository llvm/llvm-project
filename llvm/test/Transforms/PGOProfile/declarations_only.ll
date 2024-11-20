; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN --check-prefix=GEN-COMDAT

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN-COMDAT: $__llvm_profile_raw_version = comdat any
; GEN-COMDAT: @__llvm_profile_raw_version = hidden constant i64 {{[0-9]+}}, comdat
; GEN-NOT: @__profn_test_1 = private constant [6 x i8] c"test_1"
; GEN-NOT: @__profn_test_2 = private constant [6 x i8] c"test_2"

declare i32 @test_1(i32 %i)

declare i32 @test_2(i32 %i)
