; RUN: llc -march=mips -mcpu=mips32  < %s | FileCheck %s --check-prefixes=ALL,BE
; RUN: llc -march=mipsel -mcpu=mips32  < %s | FileCheck %s --check-prefixes=ALL,LE

; Verify visitTRUNCATE respects endianness when transforming trunc to insert_vector_elt.

; ALL-LABEL: a:
; BE: lw $2, 4($4)
; LE: lw $2, 0($4)

define i32 @a(ptr %a) {
entry:
%0 = load <2 x i32>, ptr %a
%1 = bitcast <2 x i32> %0 to i64
%2 = trunc i64 %1 to i32
ret i32 %2
}

; ALL-LABEL: b:
; BE: lw $2, 12($4)
; LE: lw $2, 0($4)

define i32 @b(ptr %a) {
entry:
%0 = load <4 x i32>, ptr %a
%1 = bitcast <4 x i32> %0 to i128
%2 = trunc i128 %1 to i32
ret i32 %2
}


; Verify visitEXTRACT_VECTOR_ELT respects endianness when transforming extract_vector_elt to a trunc.

; ALL-LABEL: c:
; BE: lw $2, 0($4)
; LE: lw $2, 0($4)

define i32 @c(ptr %a) {
entry:
%0 = load i64, ptr %a
%1 = bitcast i64 %0 to <2 x i32>
%2 = extractelement <2 x i32> %1, i32 0
ret i32 %2
}

; ALL-LABEL: d:
; BE: lw $2, 4($4)
; LE: lw $2, 4($4)

define i32 @d(ptr %a) {
entry:
%0 = load i64, ptr %a
%1 = bitcast i64 %0 to <2 x i32>
%2 = extractelement <2 x i32> %1, i32 1
ret i32 %2
}
