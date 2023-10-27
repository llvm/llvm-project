; RUN: llc -denormal-fp-math=dynamic --denormal-fp-math-f32=preserve-sign -stop-after=finalize-isel < %s | FileCheck %s

; Check that the command line flag annotates the IR with the
; appropriate attributes.

; CHECK: attributes #0 = { "denormal-fp-math"="dynamic,dynamic" "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
define float @foo(float %var) {
  ret float %var
}
