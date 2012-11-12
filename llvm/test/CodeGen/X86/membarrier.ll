; RUN: llc < %s -march=x86-64 -mattr=-sse -O0
; PR9675

define i32 @t() {
entry:
  %i = alloca i32, align 4
  store i32 1, i32* %i, align 4
  fence seq_cst
  %0 = atomicrmw sub i32* %i, i32 1 monotonic
  fence seq_cst
  ret i32 0
}
