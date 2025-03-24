; RUN: not llvm-ml -m64 -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s

.code

mov word ptr al, ax
; CHECK: error: cannot cast register 'AL' to 'word'; size does not match

mov dword ptr al, eax
; CHECK: error: cannot cast register 'AL' to 'dword'; size does not match

mov qword ptr al, rax
; CHECK: error: cannot cast register 'AL' to 'qword'; size does not match

mov byte ptr ax, al
; CHECK: error: cannot cast register 'AX' to 'byte'; size does not match

mov dword ptr ax, eax
; CHECK: error: cannot cast register 'AX' to 'dword'; size does not match

mov qword ptr ax, rax
; CHECK: error: cannot cast register 'AX' to 'qword'; size does not match

mov byte ptr eax, al
; CHECK: error: cannot cast register 'EAX' to 'byte'; size does not match

mov word ptr eax, ax
; CHECK: error: cannot cast register 'EAX' to 'word'; size does not match

mov qword ptr eax, rax
; CHECK: error: cannot cast register 'EAX' to 'qword'; size does not match

mov byte ptr rax, al
; CHECK: error: cannot cast register 'RAX' to 'byte'; size does not match

mov word ptr rax, ax
; CHECK: error: cannot cast register 'RAX' to 'word'; size does not match

mov dword ptr rax, eax
; CHECK: error: cannot cast register 'RAX' to 'dword'; size does not match

END
