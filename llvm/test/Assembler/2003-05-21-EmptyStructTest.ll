; RUN: llvm-as %s -o /dev/null

; The old C front-end never generated empty structures, now the new one
; can.  For some reason we never handled them in the parser. Weird.

@X = global {  } zeroinitializer
