; RUN: llc -opaque-pointers=0 -march=hexagon -debug-only=commgep 2>&1 < %s | FileCheck %s
; REQUIRES: asserts

; We should generate new GEPs with "inbounds" flag.
; CHECK: new GEP:{{.*}}inbounds
; CHECK: new GEP:{{.*}}inbounds

target triple = "hexagon"

%struct.0 = type { i16, i16 }

; Function Attrs: nounwind
define i16 @TraceBack(%struct.0* %t) #0 {
entry:
  %p = getelementptr inbounds %struct.0, %struct.0* %t, i32 0, i32 0
  %a = load i16, i16* %p
  ret i16 %a
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
