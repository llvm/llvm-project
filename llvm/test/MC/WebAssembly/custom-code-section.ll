; RUN: llc -mtriple wasm32-unknown-unknown-wasm -O2 -filetype=obj %s -o %t.o

; Wasm silently ignores custom sections for code.
; We had a bug where this cause a crash

define hidden void @call_indirect() section "some_section_name" {
entry:
  ret void
}
