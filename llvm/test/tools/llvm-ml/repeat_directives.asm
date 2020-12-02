; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.data

a BYTE ?

.code

repeat_test PROC
; CHECK-LABEL: repeat_test:
  REPEAT 1+2
    xor eax, 0
  ENDM
; CHECK: xor eax, 0
; CHECK: xor eax, 0
; CHECK: xor eax, 0
; CHECK-NOT: xor eax, 0
repeat_test ENDP

while_test PROC
; CHECK-LABEL: while_test:
  C = 1
  WHILE C <= 3
    xor eax, C
    C = C + 1
  ENDM
; CHECK: xor eax, 1
; CHECK: xor eax, 2
; CHECK: xor eax, 3
; CHECK-NOT: xor eax,
while_test ENDP

for_test PROC
; CHECK-LABEL: for_test:
  FOR arg, <'O', 'K', 13, 10>
    mov al, arg
  ENDM
; CHECK: mov al, 79
; CHECK: mov al, 75
; CHECK: mov al, 13
; CHECK: mov al, 10
; CHECK-NOT: mov al,
for_test ENDP

for_without_substitution_test PROC
; CHECK-LABEL: for_without_substitution_test:
  FOR a, <'O', 'K', 13, 10>
    mov al, 'a'
  ENDM
; CHECK: mov al, 97
; CHECK: mov al, 97
; CHECK: mov al, 97
; CHECK: mov al, 97
; CHECK-NOT: mov al,
for_without_substitution_test ENDP

for_with_default_test PROC
; CHECK-LABEL: for_with_default_test:
  FOR arg:=<'K'>, <'O', ,, 13,>
    mov al, arg
  ENDM
; CHECK: mov al, 79
; CHECK: mov al, 75
; CHECK: mov al, 75
; CHECK: mov al, 13
; CHECK: mov al, 75
; CHECK-NOT: mov al,
for_with_default_test ENDP

forc_test PROC
; CHECK-LABEL: forc_test:
  FORC arg, <OK>
    mov al, "&arg"
  ENDM
; CHECK: mov al, 79
; CHECK: mov al, 75
; CHECK-NOT: mov al,
forc_test ENDP

forc_improper_test PROC
; CHECK-LABEL: forc_improper_test:
  FORC arg, A-; OK
    mov al, "&arg"
  ENDM
; CHECK: mov al, 65
; CHECK: mov al, 45
; CHECK: mov al, 59
; CHECK-NOT: mov al,
forc_improper_test ENDP

nested_substitution_test PROC
; CHECK-LABEL: nested_substitution_test:
  FOR s, <A-, OK>
    FORC c, <s>
      mov al, '&c'
    ENDM
  ENDM
; CHECK: mov al, 65
; CHECK: mov al, 45
; CHECK: mov al, 79
; CHECK: mov al, 75
; CHECK-NOT: mov al,
nested_substitution_test ENDP

end
