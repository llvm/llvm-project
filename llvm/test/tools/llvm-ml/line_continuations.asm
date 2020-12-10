# RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.code

t1:
mov eax, \
    ebx
# CHECK: t1:
# CHECK-NEXT: mov eax, ebx

t2:
mov eax, [ebx + \
          1]
# CHECK: t2:
# CHECK-NEXT: mov eax, dword ptr [ebx + 1]

END
