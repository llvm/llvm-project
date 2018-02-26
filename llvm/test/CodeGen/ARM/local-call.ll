; RUN: llc -mtriple=thumbv7-apple-ios -filetype=obj %s -o %t
; RUN: llvm-objdump -macho -d %t | FileCheck %s

; This function just messes up the offsets enough to make the libcall in
; test_local_call unencodable with a blx.
define void @thing() {
  ret void
}

define i64 @__udivdi3(i64 %a, i64 %b) {
  ret i64 %b
}

define i64 @test_local_call(i64 %a, i64 %b) {
; CHECK-LABEL: test_local_call:
; CHECK: bl ___udivdi3

%res = udiv i64 %a, %b
  ret i64 %res
}
