; RUN: opt < %s -S -early-cse | FileCheck %s --check-prefix=CHECK-NOMEMSSA
; RUN: opt < %s -S -basicaa -early-cse-memssa | FileCheck %s
; RUN: opt < %s -S -passes='early-cse' | FileCheck %s --check-prefix=CHECK-NOMEMSSA
; RUN: opt < %s -S -aa-pipeline=basic-aa -passes='early-cse-memssa' | FileCheck %s

@G1 = global i32 zeroinitializer
@G2 = global i32 zeroinitializer

;; Simple load value numbering across non-clobbering store.
; CHECK-LABEL: @test1(
; CHECK-NOMEMSSA-LABEL: @test1(
define i32 @test1() {
  %V1 = load i32, i32* @G1
  store i32 0, i32* @G2
  %V2 = load i32, i32* @G1
  ; CHECK-NOMEMSSA: sub i32 %V1, %V2
  %Diff = sub i32 %V1, %V2
  ret i32 %Diff
  ; CHECK: ret i32 0
}

;; Simple dead store elimination across non-clobbering store.
; CHECK-LABEL: @test2(
; CHECK-NOMEMSSA-LABEL: @test2(
define void @test2() {
entry:
  %V1 = load i32, i32* @G1
  ; CHECK: store i32 0, i32* @G2
  store i32 0, i32* @G2
  ; CHECK-NOT: store
  ; CHECK-NOMEMSSA: store i32 %V1, i32* @G1
  store i32 %V1, i32* @G1
  ret void
}
