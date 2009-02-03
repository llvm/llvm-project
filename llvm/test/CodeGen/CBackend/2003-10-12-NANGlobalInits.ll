; RUN: llvm-as < %s | llc -march=c

; This is a non-normal FP value: it's a nan.
@NAN = global { float } { float 0x7FF8000000000000 }            ; <{ float }*> [#uses=0]
@NANs = global { float } { float 0x7FFC000000000000 }           ; <{ float }*> [#uses=0]
