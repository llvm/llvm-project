; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t.o
; RUN: llvm-modextract -b -n 0 %t.o -o - | llvm-dis | FileCheck %s --check-prefix=UNIT0
; RUN: llvm-modextract -b -n 1 %t.o -o - | llvm-dis | FileCheck %s --check-prefix=UNIT1

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

; After split LTO unit generation, @test_gv will move to UNIT1 (because it has a
; type metadata), and because it refers to @test, @test will be promoted by
; adding its module ID hash to its name, and .lto_set_conditional directive will
; be written in Unit 0.

; UNIT0: module asm ".lto_set_conditional test,test.{{[0-9a-f]+}}"
; UNIT0: define hidden i32 @test.{{[0-9a-f]+}}()

; Unit 1 will contain @test's declaration. The normal ThinLTO split bitcode
; writing removes the signatures from the split unit's declaration, but in Wasm
; you should not do this in order to avoid function signature mismatch in
; linker.

; UNIT1: declare hidden i32 @test.{{[0-9a-f]+}}()

@test_gv = constant ptr @test, align 4, !type !0

define internal i32 @test() {
  ret i32 0
}

!0 = !{i64 0, !{}}
