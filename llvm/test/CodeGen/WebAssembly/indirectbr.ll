; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; This tests that indirectbr instructions are lowered to switches. Currently we
; just re-use the IndirectBrExpand Pass; it has its own IR-level test.
; So this test just ensures that the pass gets run and we can lower indirectbr

target triple = "wasm32"

@test1.targets = constant [4 x i8*] [i8* blockaddress(@test1, %bb0),
                                     i8* blockaddress(@test1, %bb1),
                                     i8* blockaddress(@test1, %bb2),
                                     i8* blockaddress(@test1, %bb3)]

; Just check the barest skeleton of the structure
; CHECK-LABEL: test1:
; CHECK: i32.load
; CHECK: i32.load
; CHECK: loop
; CHECK: block
; CHECK: block
; CHECK: block
; CHECK: block
; CHECK: br_table ${{[^,]+}}, 1, 2, 0
; CHECK: end_block
; CHECK: end_block
; CHECK: end_block
; CHECK: end_block
; CHECK: br
; CHECK: end_loop
; CHECK: end_function
; CHECK: test1.targets:
; CHECK-NEXT: .int32
; CHECK-NEXT: .int32
; CHECK-NEXT: .int32
; CHECK-NEXT: .int32

define void @test1(i32* readonly %p, i32* %sink) #0 {

entry:
  %i0 = load i32, i32* %p
  %target.i0 = getelementptr [4 x i8*], [4 x i8*]* @test1.targets, i32 0, i32 %i0
  %target0 = load i8*, i8** %target.i0
  ; Only a subset of blocks are viable successors here.
  indirectbr i8* %target0, [label %bb0, label %bb1]


bb0:
  store volatile i32 0, i32* %sink
  br label %latch

bb1:
  store volatile i32 1, i32* %sink
  br label %latch

bb2:
  store volatile i32 2, i32* %sink
  br label %latch

bb3:
  store volatile i32 3, i32* %sink
  br label %latch

latch:
  %i.next = load i32, i32* %p
  %target.i.next = getelementptr [4 x i8*], [4 x i8*]* @test1.targets, i32 0, i32 %i.next
  %target.next = load i8*, i8** %target.i.next
  ; A different subset of blocks are viable successors here.
  indirectbr i8* %target.next, [label %bb1, label %bb2]
}
