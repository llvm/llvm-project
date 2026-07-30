; RUN: llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s

.code

num EQU 276
var TEXTEQU %num

ECHO t1
ECHO var
 ECHO var

; CHECK-LABEL: t1
; CHECK: var
; CHECK: var
; CHECK-NOT: var

ECHO t2
%ECHO var
% ECHO var
 %ECHO var
 % ECHO var

; CHECK-LABEL: t2
; CHECK: 276
; CHECK: 276
; CHECK: 276
; CHECK: 276
; CHECK-NOT: 276

t3:
mov eax, var
% mov eax, var

; CHECK-LABEL: t3:
; CHECK: mov eax, 276
; CHECK: mov eax, 276

end
