; RUN: not llvm-ml -m64 -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s

.code

mov word ptr al, ax
; CHECK: [[#@LINE-1]]:14: error: 8-bit register 'AL' cannot be used as a 16-bit WORD

mov dword ptr al, eax
; CHECK: [[#@LINE-1]]:15: error: 8-bit register 'AL' cannot be used as a 32-bit DWORD

mov qword ptr al, rax
; CHECK: [[#@LINE-1]]:15: error: 8-bit register 'AL' cannot be used as a 64-bit QWORD

mov byte ptr ax, al
; CHECK: [[#@LINE-1]]:14: error: 16-bit register 'AX' cannot be used as a 8-bit BYTE

mov dword ptr ax, eax
; CHECK: [[#@LINE-1]]:15: error: 16-bit register 'AX' cannot be used as a 32-bit DWORD

mov qword ptr ax, rax
; CHECK: [[#@LINE-1]]:15: error: 16-bit register 'AX' cannot be used as a 64-bit QWORD

mov byte ptr eax, al
; CHECK: [[#@LINE-1]]:14: error: 32-bit register 'EAX' cannot be used as a 8-bit BYTE

mov word ptr eax, ax
; CHECK: [[#@LINE-1]]:14: error: 32-bit register 'EAX' cannot be used as a 16-bit WORD

mov qword ptr eax, rax
; CHECK: [[#@LINE-1]]:15: error: 32-bit register 'EAX' cannot be used as a 64-bit QWORD

mov byte ptr rax, al
; CHECK: [[#@LINE-1]]:14: error: 64-bit register 'RAX' cannot be used as a 8-bit BYTE

mov word ptr rax, ax
; CHECK: [[#@LINE-1]]:14: error: 64-bit register 'RAX' cannot be used as a 16-bit WORD

mov dword ptr rax, eax
; CHECK: [[#@LINE-1]]:15: error: 64-bit register 'RAX' cannot be used as a 32-bit DWORD

END
