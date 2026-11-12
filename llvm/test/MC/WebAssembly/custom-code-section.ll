; RUN: llc -O2 -filetype=obj %s -o %t.o

target triple = "wasm32-unknown-unknown"

; Wasm silently ignores custom sections for code.
; We had a bug where this cause a crash

define hidden void @call_indirect() section "some_section_name" {
entry:
  ret void
}
