; RUN: opt -S -simplifycfg < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7a--none-eabi"

; One of the phi node's values is too complex to be represented by a global
; variable, so we can't convert to a lookup table.

; CHECK-NOT: @switch.table
; CHECK-NOT: load

@g1 = external global i32, align 4
@g2 = external global i32, align 4
@g3 = external global i32, align 4
@g4 = external thread_local global i32, align 4

define i32* @test3(i32 %n) {
entry:
  switch i32 %n, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ]

sw.bb:
  br label %return

sw.bb1:
  br label %return

sw.bb2:
  br label %return

sw.default:
  br label %return

return:
  %retval.0 = phi i32* [ @g4, %sw.default ], [ getelementptr inbounds (i32, i32* inttoptr (i32 mul (i32 ptrtoint (i32* @g3 to i32), i32 2) to i32*), i32 1), %sw.bb2 ], [ @g2, %sw.bb1 ], [ @g1, %sw.bb ]
  ret i32* %retval.0
}
