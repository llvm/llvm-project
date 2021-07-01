; RUN: llvm-ml -m32 -filetype=s %s /Fo - | FileCheck %s --check-prefix=CHECK-32
; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s --check-prefix=CHECK-64

.code
mov eax, [4]
; CHECK-32: mov eax, dword ptr [4]
; CHECK-64: mov eax, dword ptr [rip + 4]
END