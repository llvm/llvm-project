; REQUIRES: default_target
; RUN: llc -o /dev/null -O2 -time-trace -time-trace-granularity=100 -time-trace-file=%t.json
; RUN: FileCheck --input-file=%t.json %s

; CHECK: "traceEvents"

define void @f() {
  ret void
}
