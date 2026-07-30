; RUN: not llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.code

t1:
; CHECK: :[[# @LINE + 1]]:10: error: invalid decimal number
mov eax, 120b
; CHECK: :[[# @LINE + 1]]:10: error: invalid binary number
mov eax, 120y
.radix 11
; CHECK: :[[# @LINE + 1]]:10: error: invalid base-11 number
mov eax, 120b
; CHECK: :[[# @LINE + 1]]:10: error: invalid binary number
mov eax, 120y
.radix 10

t2:
; CHECK: :[[# @LINE + 1]]:10: error: invalid octal number
mov eax, 190o
; CHECK: :[[# @LINE + 1]]:10: error: invalid octal number
mov eax, 190q
.radix 13
; CHECK: :[[# @LINE + 1]]:10: error: invalid octal number
mov eax, 190o
; CHECK: :[[# @LINE + 1]]:10: error: invalid octal number
mov eax, 190q
.radix 10

t3:
; CHECK: :[[# @LINE + 1]]:10: error: invalid decimal number
mov eax, 1f0d
; CHECK: :[[# @LINE + 1]]:10: error: invalid decimal number
mov eax, 1f0t
.radix 13
; CHECK: :[[# @LINE + 1]]:10: error: invalid base-13 number
mov eax, 1f0d
; CHECK: :[[# @LINE + 1]]:10: error: invalid decimal number
mov eax, 1f0t
.radix 10

t4:
; CHECK: :[[# @LINE + 1]]:10: error: invalid decimal number
mov eax, 10e
.radix 16
.radix 10
; CHECK: :[[# @LINE + 1]]:10: error: invalid decimal number
mov eax, 10e

t5:
.radix 9
; CHECK: :[[# @LINE + 1]]:10: error: invalid base-9 number
mov eax, 9
.radix 10

END
