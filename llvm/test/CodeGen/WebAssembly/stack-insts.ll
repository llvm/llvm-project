; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-block-placement -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @foo0()
declare void @foo1()

; Tests if br_table is printed correctly with a tab.
; CHECK-LABEL: test0:
; CHECK: br_table {0, 1, 0, 1, 0}
define void @test0(i32 %n) {
entry:
  switch i32 %n, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb.1
    i32 2, label %sw.bb
    i32 3, label %sw.bb.1
  ]

sw.bb:                                            ; preds = %entry, %entry
  tail call void @foo0()
  br label %sw.epilog

sw.bb.1:                                          ; preds = %entry, %entry
  tail call void @foo1()
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb, %sw.bb.1
  ret void
}
