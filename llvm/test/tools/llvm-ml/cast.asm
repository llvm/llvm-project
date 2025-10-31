; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.code

mov byte ptr al, al
mov al, byte ptr al
; CHECK: mov al, al
; CHECK-NEXT: mov al, al

mov word ptr ax, ax
mov ax, word ptr ax
; CHECK: mov ax, ax
; CHECK-NEXT: mov ax, ax

mov dword ptr eax, eax
mov eax, dword ptr eax
; CHECK: mov eax, eax
; CHECK-NEXT: mov eax, eax

mov qword ptr rax, rax
mov rax, qword ptr rax
; CHECK: mov rax, rax
; CHECK-NEXT: mov rax, rax

END
