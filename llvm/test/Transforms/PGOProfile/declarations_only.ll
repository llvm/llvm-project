; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --implicit-check-not='__llvm_profile_raw_version'

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @test_1(i32 %i)

declare i32 @test_2(i32 %i)
