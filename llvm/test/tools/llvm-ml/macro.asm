; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.data

x1 DWORD ?
x2 DWORD ?
xa1 DWORD ?

.code

SubstitutionMacro macro a1:req, a2:=<7>
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

  SubstitutionMacro 1
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

  SubstitutionMacro 2, 8
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

substitution_test_lowercase PROC
; CHECK-LABEL: substitution_test_lowercase:

  substitutionmacro 2, 8
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
substitution_test_lowercase ENDP

substitution_test_uppercase PROC
; CHECK-LABEL: substitution_test_uppercase:

  SUBSTITUTIONMACRO 2, 8
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
substitution_test_uppercase ENDP

AmbiguousSubstitutionMacro MACRO x, y
  x&y BYTE 0
ENDM

ambiguous_substitution_test PROC
; CHECK-LABEL: ambiguous_substitution_test:

; should expand to ab BYTE 0
  AmbiguousSubstitutionMacro a, b

; CHECK: ab:
; CHECK-NOT: ay:
; CHECK-NOT: xb:
; CHECK-NOT: xy:
ambiguous_substitution_test ENDP

AmbiguousSubstitutionInStringMacro MACRO x, y
  BYTE "x&y"
ENDM

ambiguous_substitution_in_string_test PROC
; CHECK-LABEL: ambiguous_substitution_in_string_test:

; should expand to BYTE "5y"
  AmbiguousSubstitutionInStringMacro 5, 7

; CHECK: .byte 53
; CHECK-NEXT: .byte 121
; CHECK-NOT: .byte
ambiguous_substitution_in_string_test ENDP

OptionalParameterMacro MACRO a1:req, a2
  mov eax, a1
IFNB <a2>
  mov eax, a2
ENDIF
  ret
ENDM

optional_parameter_test PROC
; CHECK-LABEL: optional_parameter_test:

  OptionalParameterMacro 4
; CHECK: mov eax, 4
; CHECK: ret

  OptionalParameterMacro 5, 9
; CHECK: mov eax, 5
; CHECK: mov eax, 9
; CHECK: ret
optional_parameter_test ENDP

LocalSymbolMacro MACRO
  LOCAL a
a: ret
   jmp a
ENDM

local_symbol_test PROC
; CHECK-LABEL: local_symbol_test:

  LocalSymbolMacro
; CHECK: "??0000":
; CHECK-NEXT: ret
; CHECK-NEXT: jmp "??0000"

  LocalSymbolMacro
; CHECK: "??0001":
; CHECK-NEXT: ret
; CHECK-NEXT: jmp "??0001"
local_symbol_test ENDP

PURGE AmbiguousSubstitutionMacro, LocalSymbolMacro,
      OptionalParameterMacro

; Redefinition
LocalSymbolMacro MACRO
  LOCAL b
b: xor eax, eax
   jmp b
ENDM

purge_test PROC
; CHECK-LABEL: purge_test:

  LocalSymbolMacro
; CHECK: "??0002":
; CHECK-NEXT: xor eax, eax
; CHECK-NEXT: jmp "??0002"
purge_test ENDP

END
