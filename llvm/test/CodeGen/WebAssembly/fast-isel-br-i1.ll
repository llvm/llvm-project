; RUN: llc < %s -fast-isel -asm-verbose=false -wasm-keep-registers | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Fast-isel uses a 32-bit xor with -1 to negate i1 values, because it doesn't
; make any guarantees about the contents of the high bits of a register holding
; an i1 value. Test that when we do a `br_if` or `br_unless` with what what an
; i1 value in LLVM IR, that we only test the low bit.

; CHECK: i32.xor
; CHECK: i32.const       $push[[L0:[0-9]+]]=, 1{{$}}
; CHECK: i32.and         $push[[L1:[0-9]+]]=, $pop{{[0-9]+}}, $pop[[L0]]{{$}}
; CHECK: br_if           0, $pop[[L1]]{{$}}

; CHECK: i32.xor
; CHECK: i32.const       $push[[L2:[0-9]+]]=, 1{{$}}
; CHECK: i32.and         $push[[L3:[0-9]+]]=, $pop{{[0-9]+}}, $pop[[L2]]{{$}}
; CHECK: br_if           0, $pop[[L3]]{{$}}

define void @test() {
start:
  %0 = call i32 @return_one()
  br label %bb1

bb1:
  %1 = icmp eq i32 %0, 1
  %2 = xor i1 %1, true
  br i1 %2, label %bb2, label %bb3

bb2:
  call void @panic()
  unreachable

bb3:
  %3 = xor i1 %2, true
  br i1 %3, label %bb4, label %bb5

bb4:
  call void @panic()
  unreachable

bb5:
  ret void
}

declare i32 @return_one()
declare void @panic()
