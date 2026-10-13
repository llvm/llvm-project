; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data
sym1 dd 42
sym2 dd 43

; CHECK-LABEL: rva_data:
; CHECK: .long sym1@IMGREL
rva_data dd IMAGEREL sym1

; CHECK-LABEL: rva_data_offset:
; CHECK: .long sym1@IMGREL+4
rva_data_offset dd IMAGEREL sym1 + 4

; CHECK-LABEL: rva_data_paren:
; CHECK: .long sym1@IMGREL+4
rva_data_paren dd (IMAGEREL sym1) + 4

MY_STRUCT STRUCT
  field_default dd IMAGEREL sym1
MY_STRUCT ENDS

; CHECK-LABEL: struct_inst_default:
; CHECK: .long sym1@IMGREL
struct_inst_default MY_STRUCT <>

; CHECK-LABEL: struct_inst_override:
; CHECK: .long sym2@IMGREL
struct_inst_override MY_STRUCT <IMAGEREL sym2>

.code
; CHECK-LABEL: t1:
; CHECK: mov eax, offset sym1@IMGREL
t1:
mov eax, IMAGEREL sym1

; CHECK-LABEL: t2:
; CHECK: mov eax, offset sym1@IMGREL+4
t2:
mov eax, IMAGEREL sym1 + 4

; CHECK-LABEL: t3:
; CHECK: mov ebx, dword ptr [eax + sym1@IMGREL]
t3:
mov ebx, [eax + IMAGEREL sym1]

; CHECK-LABEL: t4:
; CHECK: mov eax, dword ptr fs:[sym1@IMGREL]
t4:
mov eax, fs:[IMAGEREL sym1]

; Test negative offset.
; CHECK-LABEL: t5:
; CHECK: mov eax, offset sym1@IMGREL-4
t5:
mov eax, IMAGEREL sym1 - 4

END
