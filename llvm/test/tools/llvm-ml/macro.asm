; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.data

x1 DWORD ?
x2 DWORD ?
xa1 DWORD ?

.code

substitution_macro macro a1:req, a2:=<7>
  mov eax, a1
  mov eax, a1&
  mov eax, &a1
  mov eax, &a1&

  mov eax, xa1
  mov eax, x&a1
  mov eax, x&a1&

  mov eax, a2
  mov eax, a2&
  mov eax, &a2
  mov eax, &a2&
endm

substitution_test_with_default PROC
; CHECK-LABEL: substitution_test_with_default:

  substitution_macro 1
; CHECK: mov eax, 1
; CHECK-NEXT: mov eax, 1
; CHECK-NEXT: mov eax, 1
; CHECK-NEXT: mov eax, 1
; CHECK: mov eax, dword ptr [rip + xa1]
; CHECK-NEXT: mov eax, dword ptr [rip + x1]
; CHECK-NEXT: mov eax, dword ptr [rip + x1]
; CHECK: mov eax, 7
; CHECK-NEXT: mov eax, 7
; CHECK-NEXT: mov eax, 7
; CHECK-NEXT: mov eax, 7

  ret
substitution_test_with_default ENDP

substitution_test_with_value PROC
; CHECK-LABEL: substitution_test_with_value:

  substitution_macro 2, 8
; CHECK: mov eax, 2
; CHECK-NEXT: mov eax, 2
; CHECK-NEXT: mov eax, 2
; CHECK-NEXT: mov eax, 2
; CHECK: mov eax, dword ptr [rip + xa1]
; CHECK-NEXT: mov eax, dword ptr [rip + x2]
; CHECK-NEXT: mov eax, dword ptr [rip + x2]
; CHECK: mov eax, 8
; CHECK-NEXT: mov eax, 8
; CHECK-NEXT: mov eax, 8
; CHECK-NEXT: mov eax, 8

  ret
substitution_test_with_value ENDP

ambiguous_substitution_macro MACRO x, y
  x&y BYTE 0
ENDM

ambiguous_substitution_test PROC
; CHECK-LABEL: ambiguous_substitution_test:

; should expand to ab BYTE 0
  ambiguous_substitution_macro a, b

; CHECK: ab:
; CHECK-NOT: ay:
; CHECK-NOT: xb:
; CHECK-NOT: xy:
ambiguous_substitution_test ENDP

ambiguous_substitution_in_string_macro MACRO x, y
  BYTE "x&y"
ENDM

ambiguous_substitution_in_string_test PROC
; CHECK-LABEL: ambiguous_substitution_in_string_test:

; should expand to BYTE "5y"
  ambiguous_substitution_in_string_macro 5, 7

; CHECK: .byte 53
; CHECK-NEXT: .byte 121
; CHECK-NOT: .byte
ambiguous_substitution_in_string_test ENDP

optional_parameter_macro MACRO a1:req, a2
  mov eax, a1
IFNB <a2>
  mov eax, a2
ENDIF
  ret
ENDM

optional_parameter_test PROC
; CHECK-LABEL: optional_parameter_test:

  optional_parameter_macro 4
; CHECK: mov eax, 4
; CHECK: ret

  optional_parameter_macro 5, 9
; CHECK: mov eax, 5
; CHECK: mov eax, 9
; CHECK: ret
optional_parameter_test ENDP

local_symbol_macro MACRO
  LOCAL a
a: ret
   jmp a
ENDM

local_symbol_test PROC
; CHECK-LABEL: local_symbol_test:

  local_symbol_macro
; CHECK: "??0000":
; CHECK-NEXT: ret
; CHECK-NEXT: jmp "??0000"

  local_symbol_macro
; CHECK: "??0001":
; CHECK-NEXT: ret
; CHECK-NEXT: jmp "??0001"
local_symbol_test ENDP

PURGE ambiguous_substitution_macro, local_symbol_macro,
      optional_parameter_macro

; Redefinition
local_symbol_macro MACRO
  LOCAL b
b: xor eax, eax
   jmp b
ENDM

purge_test PROC
; CHECK-LABEL: purge_test:

  local_symbol_macro
; CHECK: "??0002":
; CHECK-NEXT: xor eax, eax
; CHECK-NEXT: jmp "??0002"
purge_test ENDP

END
