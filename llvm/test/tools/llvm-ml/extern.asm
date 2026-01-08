; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s --check-prefixes=CHECK,CHECK-32
; RUN: llvm-ml64 -filetype=s %s /Fo - | FileCheck %s --check-prefixes=CHECK,CHECK-64

extern foo : dword, bar : word, baz : proc
; CHECK: .extern foo
; CHECK: .extern bar
; CHECK: .extern baz

extrn quux : dword
; CHECK: .extern quux

.code
mov ebx, foo
; CHECK-32: mov ebx, dword ptr [foo]
; CHECK-64: mov ebx, dword ptr [rip + foo]

mov bx, bar
; CHECK-32: mov bx, word ptr [bar]
; CHECK-64: mov bx, word ptr [rip + bar]

mov edx, quux
; CHECK-32: mov edx, dword ptr [quux]
; CHECK-64: mov edx, dword ptr [rip + quux]

END
