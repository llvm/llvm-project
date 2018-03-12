; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts
; This should compile cleanly.

target triple = "hexagon"

%s.0 = type opaque

@g0 = external global %s.0

; Function Attrs: nounwind
define %s.0* @f0() #0 {
b0:
  ret %s.0* @g0
}

attributes #0 = { nounwind }
