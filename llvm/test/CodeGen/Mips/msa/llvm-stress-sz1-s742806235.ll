; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed to select code for a truncstore of a
; build_vector.
; It should at least successfully build.

define void @autogen_SD742806235(ptr, ptr, ptr, i32, i64, i8) {
BB:
  %A4 = alloca double
  %A3 = alloca double
  %A2 = alloca <8 x i8>
  %A1 = alloca <4 x float>
  %A = alloca i1
  store i8 %5, ptr %0
  store i8 %5, ptr %0
  store i8 %5, ptr %0
  store <8 x i8> <i8 0, i8 -1, i8 0, i8 -1, i8 0, i8 -1, i8 0, i8 -1>, ptr %A2
  store i8 %5, ptr %0
  ret void
}
