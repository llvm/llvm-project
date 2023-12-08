; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data

align_test:
ALIGN 16
; CHECK-LABEL: align_test:
; CHECK-NEXT: .p2align 4

org_test:
ORG 256
; CHECK-LABEL: org_test:
; CHECK-NEXT: .org 256, 0

align_struct STRUCT
  BYTE ?

  ALIGN 4
  x BYTE ?
  x_succ BYTE ?
  BYTE ?

  ALIGN 2
  y BYTE ?
  y_succ BYTE ?

  ALIGN 1
  z BYTE ?

  EVEN
  q BYTE ?
align_struct ENDS

struct_align_data ALIGN_STRUCT <101, 102, 103, 104, 105, 106, 107, 108>
; CHECK-LABEL: struct_align_data:
; CHECK-NEXT: .byte 101
; CHECK-NEXT: .zero 3
; CHECK-NEXT: .byte 102
; CHECK-NEXT: .byte 103
; CHECK-NEXT: .byte 104
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .byte 105
; CHECK-NEXT: .byte 106
; CHECK-NEXT: .byte 107
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .byte 108

org_struct STRUCT
  x BYTE ?
  x_succ BYTE ?
  ORG 15
  y BYTE ?
  y_succ BYTE ?
  ORG 2
  z BYTE ?
  z_succ BYTE ?
org_struct ENDS

.code

struct_align_test PROC

x_align_test:
  MOV eax, align_struct.x
  MOV eax, align_struct.x_succ
; CHECK-LABEL: x_align_test:
; CHECK-NEXT: mov eax, 4
; CHECK-NEXT: mov eax, 5

y_align_test:
  MOV eax, align_struct.y
  MOV eax, align_struct.y_succ
; CHECK-LABEL: y_align_test:
; CHECK-NEXT: mov eax, 8
; CHECK-NEXT: mov eax, 9

z_align_test:
  MOV eax, align_struct.z
; CHECK-LABEL: z_align_test:
; CHECK-NEXT: mov eax, 10

q_even_test:
  MOV eax, align_struct.q
; CHECK-LABEL: q_even_test:
; CHECK-NEXT: mov eax, 12

size_align_test:
  MOV eax, SIZEOF(align_struct)
; CHECK-LABEL: size_align_test:
; CHECK-NEXT: mov eax, 13

  ret
struct_align_test ENDP

struct_org_test PROC
; CHECK-LABEL: struct_org_test:

field_positions:
  MOV eax, org_struct.x
  MOV eax, org_struct.x_succ
  MOV eax, org_struct.y
  MOV eax, org_struct.y_succ
  MOV eax, org_struct.z
  MOV eax, org_struct.z_succ
; CHECK-LABEL: field_positions:
; CHECK-NEXT: mov eax, 0
; CHECK-NEXT: mov eax, 1
; CHECK-NEXT: mov eax, 15
; CHECK-NEXT: mov eax, 16
; CHECK-NEXT: mov eax, 2
; CHECK-NEXT: mov eax, 3

  ret
struct_org_test ENDP

end
