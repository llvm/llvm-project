; RUN: llc -mtriple=hexagon -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s
; Check that we generate dual stores in one packet in V4

; CHECK: 00 40 9f 52 529f4000
; CHECK: 10 10 00 f0 f0001010

define void @foo(ptr %a, ptr %b) {
  store i32 0, ptr %a
  store i32 0, ptr %b
  ret void
}
