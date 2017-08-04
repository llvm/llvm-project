; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -verify-machineinstrs | FileCheck %s

; Test that phis are lowered.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; Basic phi triangle.

; CHECK-LABEL: test0:
; CHECK: return $0
; CHECK: div_s $push[[NUM0:[0-9]+]]=, $0, $pop[[NUM1:[0-9]+]]{{$}}
; CHECK: return $pop[[NUM0]]{{$}}
define i32 @test0(i32 %p) {
entry:
  %t = icmp slt i32 %p, 0
  br i1 %t, label %true, label %done
true:
  %a = sdiv i32 %p, 3
  br label %done
done:
  %s = phi i32 [ %a, %true ], [ %p, %entry ]
  ret i32 %s
}

; Swap phis.

; CHECK-LABEL: test1:
; CHECK: .LBB1_1:
; CHECK: copy_local $[[NUM0:[0-9]+]]=, $[[NUM1:[0-9]+]]{{$}}
; CHECK: copy_local $[[NUM1]]=, $[[NUM2:[0-9]+]]{{$}}
; CHECK: copy_local $[[NUM2]]=, $[[NUM0]]{{$}}
define i32 @test1(i32 %n) {
entry:
  br label %loop

loop:
  %a = phi i32 [ 0, %entry ], [ %b, %loop ]
  %b = phi i32 [ 1, %entry ], [ %a, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  %i.next = add i32 %i, 1
  %t = icmp slt i32 %i.next, %n
  br i1 %t, label %loop, label %exit

exit:
  ret i32 %a
}
