; RUN: llc -march=hexagon -O3 < %s
; REQUIRES: asserts

; Test that the compiler doesn't assert when attempting to
; generate a store absolute set insturction where the base
; register and destination register are same.

target triple = "hexagon-unknown--elf"

%s.0 = type { %s.1, %s.2 }
%s.1 = type { ptr, ptr }
%s.2 = type { %s.3 }
%s.3 = type { %s.4 }
%s.4 = type { %s.5, i32, i32, ptr }
%s.5 = type { i32 }

@g0 = external global %s.0, align 4

; Function Attrs: nounwind
define void @f0() #0 section ".init.text" {
b0:
  store ptr @g0, ptr @g0, align 4
  store ptr @g0, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 0, i32 1), align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
