; RUN: not llvm-ml -m64 -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s

.code

mov word ptr al, ax
; CHECK: error: 8-bit register 'AL' cannot be used as a 16-bit WORD

mov dword ptr al, eax
; CHECK: error: 8-bit register 'AL' cannot be used as a 32-bit DWORD

mov qword ptr al, rax
; CHECK: error: 8-bit register 'AL' cannot be used as a 64-bit QWORD

mov byte ptr ax, al
; CHECK: error: 16-bit register 'AX' cannot be used as a 8-bit BYTE

mov dword ptr ax, eax
; CHECK: error: 16-bit register 'AX' cannot be used as a 32-bit DWORD

mov qword ptr ax, rax
; CHECK: error: 16-bit register 'AX' cannot be used as a 64-bit QWORD

mov byte ptr eax, al
; CHECK: error: 32-bit register 'EAX' cannot be used as a 8-bit BYTE

mov word ptr eax, ax
; CHECK: error: 32-bit register 'EAX' cannot be used as a 16-bit WORD

mov qword ptr eax, rax
; CHECK: error: 32-bit register 'EAX' cannot be used as a 64-bit QWORD

mov byte ptr rax, al
; CHECK: error: 64-bit register 'RAX' cannot be used as a 8-bit BYTE

mov word ptr rax, ax
; CHECK: error: 64-bit register 'RAX' cannot be used as a 16-bit WORD

mov dword ptr rax, eax
; CHECK: error: 64-bit register 'RAX' cannot be used as a 32-bit DWORD

END
