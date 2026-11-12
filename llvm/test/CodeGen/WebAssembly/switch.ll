; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -disable-block-placement -verify-machineinstrs | FileCheck %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -disable-block-placement -verify-machineinstrs | FileCheck %s

; Test switch instructions. Block placement is disabled because it reorders
; the blocks in a way that isn't interesting here.

declare void @foo0()
declare void @foo1()
declare void @foo2()
declare void @foo3()
declare void @foo4()
declare void @foo5()

; CHECK-LABEL: bar32:
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK-NEXT: br_table {{[^,]+}}, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6{{$}}
; CHECK:      .LBB{{[0-9]+}}_1:
; CHECK:        call foo0{{$}}
; CHECK:      .LBB{{[0-9]+}}_2:
; CHECK:        call foo1{{$}}
; CHECK:      .LBB{{[0-9]+}}_3:
; CHECK:        call foo2{{$}}
; CHECK:      .LBB{{[0-9]+}}_4:
; CHECK:        call foo3{{$}}
; CHECK:      .LBB{{[0-9]+}}_5:
; CHECK:        call foo4{{$}}
; CHECK:      .LBB{{[0-9]+}}_6:
; CHECK:        call foo5{{$}}
; CHECK:      .LBB{{[0-9]+}}_7:
; CHECK:        return{{$}}
define void @bar32(i32 %n) {
entry:
  switch i32 %n, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb
    i32 2, label %sw.bb
    i32 3, label %sw.bb
    i32 4, label %sw.bb
    i32 5, label %sw.bb
    i32 6, label %sw.bb
    i32 7, label %sw.bb.1
    i32 8, label %sw.bb.1
    i32 9, label %sw.bb.1
    i32 10, label %sw.bb.1
    i32 11, label %sw.bb.1
    i32 12, label %sw.bb.1
    i32 13, label %sw.bb.1
    i32 14, label %sw.bb.1
    i32 15, label %sw.bb.2
    i32 16, label %sw.bb.2
    i32 17, label %sw.bb.2
    i32 18, label %sw.bb.2
    i32 19, label %sw.bb.2
    i32 20, label %sw.bb.2
    i32 21, label %sw.bb.3
    i32 22, label %sw.bb.4
    i32 23, label %sw.bb.5
  ]

sw.bb:                                            ; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry
  tail call void @foo0()
  br label %sw.epilog

sw.bb.1:                                          ; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
  tail call void @foo1()
  br label %sw.epilog

sw.bb.2:                                          ; preds = %entry, %entry, %entry, %entry, %entry, %entry
  tail call void @foo2()
  br label %sw.epilog

sw.bb.3:                                          ; preds = %entry
  tail call void @foo3()
  br label %sw.epilog

sw.bb.4:                                          ; preds = %entry
  tail call void @foo4()
  br label %sw.epilog

sw.bb.5:                                          ; preds = %entry
  tail call void @foo5()
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb.5, %sw.bb.4, %sw.bb.3, %sw.bb.2, %sw.bb.1, %sw.bb
  ret void
}

; CHECK-LABEL: bar64:
; CHECK:      block   {{$}}
; CHECK:      i64.const
; CHECK:      i64.gt_u
; CHECK:      br_if 0
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      block   {{$}}
; CHECK:      i32.wrap_i64
; CHECK-NEXT: br_table {{[^,]+}}, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5, 0{{$}}
; CHECK:      .LBB{{[0-9]+}}_2:
; CHECK:        call foo0{{$}}
; CHECK:      .LBB{{[0-9]+}}_3:
; CHECK:        call foo1{{$}}
; CHECK:      .LBB{{[0-9]+}}_4:
; CHECK:        call foo2{{$}}
; CHECK:      .LBB{{[0-9]+}}_5:
; CHECK:        call foo3{{$}}
; CHECK:      .LBB{{[0-9]+}}_6:
; CHECK:        call foo4{{$}}
; CHECK:      .LBB{{[0-9]+}}_7:
; CHECK:        call foo5{{$}}
; CHECK:      .LBB{{[0-9]+}}_8:
; CHECK:        return{{$}}
define void @bar64(i64 %n) {
entry:
  switch i64 %n, label %sw.epilog [
    i64 0, label %sw.bb
    i64 1, label %sw.bb
    i64 2, label %sw.bb
    i64 3, label %sw.bb
    i64 4, label %sw.bb
    i64 5, label %sw.bb
    i64 6, label %sw.bb
    i64 7, label %sw.bb.1
    i64 8, label %sw.bb.1
    i64 9, label %sw.bb.1
    i64 10, label %sw.bb.1
    i64 11, label %sw.bb.1
    i64 12, label %sw.bb.1
    i64 13, label %sw.bb.1
    i64 14, label %sw.bb.1
    i64 15, label %sw.bb.2
    i64 16, label %sw.bb.2
    i64 17, label %sw.bb.2
    i64 18, label %sw.bb.2
    i64 19, label %sw.bb.2
    i64 20, label %sw.bb.2
    i64 21, label %sw.bb.3
    i64 22, label %sw.bb.4
    i64 23, label %sw.bb.5
  ]

sw.bb:                                            ; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry
  tail call void @foo0()
  br label %sw.epilog

sw.bb.1:                                          ; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
  tail call void @foo1()
  br label %sw.epilog

sw.bb.2:                                          ; preds = %entry, %entry, %entry, %entry, %entry, %entry
  tail call void @foo2()
  br label %sw.epilog

sw.bb.3:                                          ; preds = %entry
  tail call void @foo3()
  br label %sw.epilog

sw.bb.4:                                          ; preds = %entry
  tail call void @foo4()
  br label %sw.epilog

sw.bb.5:                                          ; preds = %entry
  tail call void @foo5()
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb.5, %sw.bb.4, %sw.bb.3, %sw.bb.2, %sw.bb.1, %sw.bb
  ret void
}

; CHECK-LABEL: truncated:
; CHECK:        block
; CHECK:        block
; CHECK:        block
; CHECK:        i32.wrap_i64
; CHECK-NEXT:   br_table {{[^,]+}}, 0, 1, 2{{$}}
; CHECK:      .LBB{{[0-9]+}}_1
; CHECK:        end_block
; CHECK:        call foo0{{$}}
; CHECK:        return{{$}}
; CHECK:      .LBB{{[0-9]+}}_2
; CHECK:        end_block
; CHECK:        call foo1{{$}}
; CHECK:        return{{$}}
; CHECK:      .LBB{{[0-9]+}}_3
; CHECK:        end_block
; CHECK:        call foo2{{$}}
; CHECK:        return{{$}}
; CHECK:        end_function
define void @truncated(i64 %n) {
entry:
  %m = trunc i64 %n to i32
  switch i32 %m, label %default [
    i32 0, label %bb1
    i32 1, label %bb2
  ]

bb1:
  tail call void @foo0()
  ret void

bb2:
  tail call void @foo1()
  ret void

default:
  tail call void @foo2()
  ret void
}
