; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.code

identity MACRO arg
  exitm <arg>
endm

argument_test PROC
; CHECK-LABEL: argument_test:

  mov eax, identity(2)
; CHECK: mov eax, 2

  ret
argument_test ENDP

argument_with_parens_test PROC
; CHECK-LABEL: argument_with_parens_test:

  mov eax, identity((3))
; CHECK: mov eax, 3
  mov eax, identity(((4-1)-1))
; CHECK: mov eax, 2

  ret
argument_with_parens_test ENDP

offsetof MACRO structure, field
  EXITM <structure.&field>
ENDM

S1 STRUCT
  W byte 0
  X byte 0
  Y byte 0
S1 ENDS

substitutions_test PROC
; CHECK-LABEL: substitutions_test:

  mov eax, offsetof(S1, X)
; CHECK: mov eax, 1
  mov eax, offsetof(S1, Y)
; CHECK: mov eax, 2

  ret
substitutions_test ENDP

repeated_invocations_test PROC
; CHECK-LABEL: repeated_invocations_test:

  mov eax, identity(identity(1))
; CHECK: mov eax, 1

  ret
repeated_invocations_test ENDP

factorial MACRO n
  IF n LE 1
    EXITM <(1)>
  ELSE
    EXITM <(n)*factorial(n-1)>
  ENDIF
ENDM

; NOTE: This version is more sensitive to unintentional end-of-statement tokens.
factorial2 MACRO n
  IF n LE 1
    EXITM <(1)>
  ELSE
    EXITM <(n)*(factorial(n-1))>
  ENDIF
ENDM

string_recursive_test PROC
; CHECK-LABEL: string_recursive_test:

  mov eax, factorial(5)
; CHECK: mov eax, 120
  mov eax, factorial2(4)
; CHECK: mov eax, 24
  mov eax, 11 + factorial(6) - 11
; CHECK: mov eax, 720

  ret
string_recursive_test ENDP

fibonacci MACRO n
  IF n LE 2
    EXITM %1
  ELSE
    EXITM %fibonacci(n-1)+fibonacci(n-2)
  ENDIF
ENDM

expr_recursive_test PROC
; CHECK-LABEL: expr_recursive_test:

  mov eax, fibonacci(10)
; CHECK: mov eax, 55

  ret
expr_recursive_test ENDP

custom_strcat MACRO arg1, arg2
  EXITM <arg1&arg2>
ENDM

expand_as_directive_test custom_strcat(P, ROC)
; CHECK-LABEL: expand_as_directive_test:

  ret
expand_as_directive_test ENDP

end
