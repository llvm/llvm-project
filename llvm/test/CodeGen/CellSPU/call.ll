; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep brsl    %t1.s | count 1
; RUN: grep brasl   %t1.s | count 1
; RUN: grep stqd    %t1.s | count 80

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i32 @main() {
entry:
  %a = call i32 @stub_1(i32 1, float 0x400921FA00000000)
  call void @extern_stub_1(i32 %a, i32 4)
  ret i32 %a
}

declare void @extern_stub_1(i32, i32)

define i32 @stub_1(i32 %x, float %y) {
entry:
  ret i32 0
}

; vararg call: ensure that all caller-saved registers are spilled to the
; stack:
define i32 @stub_2(...) {
entry:
  ret i32 0
}
