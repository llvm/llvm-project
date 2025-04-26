; RUN: llc < %s -mtriple=nvptx64 -O0 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -O0 | %ptxas-verify %}


; CHECK: .visible .func  (.param .align 128 .b8 func_retval0[256]) repro()
define <64 x i32> @repro() {

  ; CHECK: .param .align 128 .b8 retval0[256];
  %1 = tail call <64 x i32> @test(i32 0)
  ret <64 x i32> %1
}

; Function Attrs: nounwind
declare <64 x i32> @test(i32)
