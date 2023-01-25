; RUN: llc -mtriple nvptx64-nvidia-cuda -stop-after machine-cp -o - < %s 2>&1 | FileCheck %s

; Check that convergent calls are emitted using convergent MIR instructions,
; while non-convergent calls are not.

target triple = "nvptx64-nvidia-cuda"

declare void @conv() convergent
declare void @not_conv()

define void @test(ptr %f) {
  ; CHECK: ConvergentCallUniPrintCall
  ; CHECK-NEXT: @conv
  call void @conv()

  ; CHECK: CallUniPrintCall
  ; CHECK-NEXT: @not_conv
  call void @not_conv()

  ; CHECK: ConvergentCallPrintCall
  call void %f() convergent

  ; CHECK: CallPrintCall
  call void %f()

  ret void
}
