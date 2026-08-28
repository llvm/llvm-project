; A pipeline truncated by -stop-after never runs the AsmPrinter, which is what
; normally initializes the TargetLoweringObjectFile. ISel still needs it to
; mangle parameter symbols.
; RUN: llc -mtriple=nvptx64 -enable-new-pm -stop-after=finalize-isel -o - %s | FileCheck %s

; CHECK: name: test
; CHECK: LD_i32 {{.*}}&test_param_0
define i32 @test(i32 %a) {
  ret i32 %a
}
