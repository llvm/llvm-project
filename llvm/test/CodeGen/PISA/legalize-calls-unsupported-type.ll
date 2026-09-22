; RUN: not llc -mtriple=pisa -stop-after=pisa-legalize-calls -filetype=null \
; RUN:     %s -o /dev/null 2>&1 | FileCheck %s

target triple = "pisa"

define i24 @unsupported_call_type(i24 %value) {
  ret i24 %value
}

; CHECK: LLVM ERROR: unsupported PISA call integer type
