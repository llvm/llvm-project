; A struct value initialized with '<...>' must not leak the angle-bracket nesting
; depth. parseStructInitializer used to increment AngleBracketDepth twice (once in
; parseOptionalAngleBracketOpen() and once directly) while parseAngleBracketClose()
; decrements it only once. A leaked depth makes a later 'GT' operator -- which the
; expression lexer treats as '>' -- be misparsed as a closing angle bracket, so an
; 'IF <expr> GT <n>' after a '<...>' initializer failed with "expected newline".
; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data
POINT STRUCT
  x DWORD ?
POINT ENDS

p POINT <>

.code
foo:
IF 1 GT 2
  xor eax, eax
ELSE
  mov eax, 17
ENDIF
  ret

; CHECK-LABEL: foo:
; CHECK: mov eax, 17
; CHECK-NOT: xor eax, eax
