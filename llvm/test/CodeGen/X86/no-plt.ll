; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu \
; RUN:   | FileCheck -check-prefix=X64 %s

define i32 @main() #0 {
; X64: callq *_Z3foov@GOTPCREL(%rip)
; X64: callq _Z3barv

entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call1 = call i32 @_Z3foov()
  %call2 = call i32 @_Z3barv()
  ret i32 0
}

; Function Attrs: nonlazybind
declare i32 @_Z3foov() #1

declare i32 @_Z3barv() #2

attributes #1 = { nonlazybind }
