; RUN: llvm-ml -m32 -filetype=s %s /Fo - | FileCheck %s --check-prefixes=CHECK,CHECK-32
; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s --check-prefixes=CHECK,CHECK-64

.data
foo DWORD 28

bar:
DWORD 29

.code

t1:
mov eax, foo
; CHECK-LABEL: t1:
; CHECK-32: mov eax, dword ptr [foo]
; CHECK-64: mov eax, dword ptr [rip + foo]

t2:
mov eax, [foo]
; CHECK-LABEL: t2:
; CHECK-32: mov eax, dword ptr [foo]
; CHECK-64: mov eax, dword ptr [rip + foo]

t3:
mov eax, [foo+2]
; CHECK-LABEL: t3:
; CHECK-32: mov eax, dword ptr [foo+2]
; CHECK-64: mov eax, dword ptr [rip + foo+2]

t4:
mov eax, [2+foo]
; CHECK-LABEL: t4:
; CHECK-32: mov eax, dword ptr [foo+2]
; CHECK-64: mov eax, dword ptr [rip + foo+2]

t5:
mov eax, [4]
; CHECK-LABEL: t5:
; CHECK: mov eax, dword ptr [4]

t6:
mov eax, [foo+ebx]
; CHECK-LABEL: t6:
; CHECK: mov eax, dword ptr [ebx + foo]

t7:
mov eax, [bar]
; CHECK-LABEL: t7:
; CHECK: mov eax, dword ptr [bar]

t8:
mov eax, [t8]
; CHECK-LABEL: t8:
; CHECK: mov eax, dword ptr [t8]

t9:
mov eax, dword ptr [bar]
; CHECK-LABEL: t9:
; CHECK-32: mov eax, dword ptr [bar]
; CHECK-64: mov eax, dword ptr [rip + bar]

t10:
mov ebx, dword ptr [4*eax]
; CHECK: mov ebx, dword ptr [4*eax]

END
