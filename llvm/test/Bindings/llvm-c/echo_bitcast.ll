; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Regression coverage: bitcast instructions should round-trip through echo.

define float @echo_bitcast_i32_to_f32(i32 %x) {
entry:
  %bc = bitcast i32 %x to float
  ret float %bc
}

define i32 @echo_bitcast_f32_to_i32(float %x) {
entry:
  %bc = bitcast float %x to i32
  ret i32 %bc
}
