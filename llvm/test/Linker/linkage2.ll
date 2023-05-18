; RUN: llvm-link %s %p/Inputs/linkage2.ll -S | FileCheck %s
; RUN: llvm-link %p/Inputs/linkage2.ll %s -S | FileCheck %s

@test1_a = common global i8 0
; CHECK-DAG: @test1_a = common global i8 0

@test2_a = global i8 0
; CHECK-DAG: @test2_a = global i8 0

@test3_a = common global i8 0
; CHECK-DAG: @test3_a = common global i16 0

@test4_a = common global i8 0, align 8
; CHECK-DAG: @test4_a = common global i16 0, align 8
