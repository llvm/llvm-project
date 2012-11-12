; RUN: opt < %s -constprop -S | FileCheck %s
; PR2165

define <1 x i64> @test1() {
  %A = bitcast i64 63 to <1 x i64>
  ret <1 x i64> %A
; CHECK: @test1
; CHECK: ret <1 x i64> <i64 63>
}

