; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data

dq_single_character BYTE "a"
; CHECK-LABEL: dq_single_character:
; CHECK-NEXT: .byte 97
; CHECK-NOT: .byte

dq_join BYTE "ab", "cd"
; CHECK-LABEL: dq_join:
; CHECK-NEXT: .byte 97
; CHECK-NEXT: .byte 98
; CHECK-NEXT: .byte 99
; CHECK-NEXT: .byte 100
; CHECK-NOT: .byte

dq_quote_escape BYTE "ab""""cd"
; Intended result: ab""cd
; CHECK-LABEL: dq_quote_escape:
; CHECK-NEXT: .byte 97
; CHECK-NEXT: .byte 98
; CHECK-NEXT: .byte 34
; CHECK-NEXT: .byte 34
; CHECK-NEXT: .byte 99
; CHECK-NEXT: .byte 100
; CHECK-NOT: .byte

dq_single_quote BYTE "ab''''cd"
; Intended result: ab''''cd
; CHECK-LABEL: dq_single_quote:
; CHECK-NEXT: .byte 97
; CHECK-NEXT: .byte 98
; CHECK-NEXT: .byte 39
; CHECK-NEXT: .byte 39
; CHECK-NEXT: .byte 39
; CHECK-NEXT: .byte 39
; CHECK-NEXT: .byte 99
; CHECK-NEXT: .byte 100
; CHECK-NOT: .byte

sq_single_character BYTE 'a'
; CHECK-LABEL: sq_single_character:
; CHECK-NEXT: .byte 97
; CHECK-NOT: .byte

sq_join BYTE 'ab', 'cd'
; CHECK-LABEL: sq_join:
; CHECK-NEXT: .byte 97
; CHECK-NEXT: .byte 98
; CHECK-NEXT: .byte 99
; CHECK-NEXT: .byte 100
; CHECK-NOT: .byte

sq_quote_escape BYTE 'ab''''cd'
; Intended result: ab''cd
; CHECK-LABEL: sq_quote_escape:
; CHECK-NEXT: .byte 97
; CHECK-NEXT: .byte 98
; CHECK-NEXT: .byte 39
; CHECK-NEXT: .byte 39
; CHECK-NEXT: .byte 99
; CHECK-NEXT: .byte 100
; CHECK-NOT: .byte

sq_double_quote BYTE 'ab""""cd'
; Intended result: ab""""cd
; CHECK-LABEL: sq_double_quote:
; CHECK-NEXT: .byte 97
; CHECK-NEXT: .byte 98
; CHECK-NEXT: .byte 34
; CHECK-NEXT: .byte 34
; CHECK-NEXT: .byte 34
; CHECK-NEXT: .byte 34
; CHECK-NEXT: .byte 99
; CHECK-NEXT: .byte 100
; CHECK-NOT: .byte

mixed_quotes_join BYTE "a'b", 'c"d'
; Intended result: a'bc"d
; CHECK-LABEL: mixed_quotes_join:
; CHECK-NEXT: .byte 97
; CHECK-NEXT: .byte 39
; CHECK-NEXT: .byte 98
; CHECK-NEXT: .byte 99
; CHECK-NEXT: .byte 34
; CHECK-NEXT: .byte 100
; CHECK-NOT: .byte

.code

sq_char_test PROC
; CHECK-LABEL: sq_char_test:

  mov eax, 'a'
; CHECK: mov eax, 97

  mov eax, ''''
; CHECK: mov eax, 39

  mov eax, '"'
; CHECK: mov eax, 34

  ret
sq_char_test ENDP

dq_char_test PROC
; CHECK-LABEL: dq_char_test:

  mov eax, "b"
; CHECK: mov eax, 98

  mov eax, """"
; CHECK: mov eax, 34

  mov eax, "'"
; CHECK: mov eax, 39

  ret
dq_char_test ENDP

string_constant_test PROC
; CHECK-LABEL: string_constant_test:

  mov eax, 'ab'
  mov eax, "ab"
; CHECK: mov eax, 24930
; CHECK: mov eax, 24930

  mov eax, "abc"
  mov eax, 'abc'
; CHECK: mov eax, 6382179
; CHECK: mov eax, 6382179

  mov eax, "abc"""
  mov eax, 'abc'''
; CHECK: mov eax, 1633837858
; CHECK: mov eax, 1633837863

  ret
string_constant_test ENDP

end
