; RUN: llc -march=mips < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
; RUN: llc -march=mipsel < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s

; RUN-TODO: llc -march=mips64 -target-abi o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
; RUN-TODO: llc -march=mips64el -target-abi o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s

; RUN: llc -march=mips64 -target-abi n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s
; RUN: llc -march=mips64el -target-abi n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s

; RUN: llc -march=mips64 -target-abi n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s
; RUN: llc -march=mips64el -target-abi n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s

; Test the memory layout for all ABI's and byte orders as specified by section
; 4 of MD00305 (MIPS ABIs Described).
; Bitfields are not covered since they are not available as a type in LLVM IR.
;
; The assembly directives deal with endianness so we don't need to account for
; that.

; Deliberately request alignments that are too small for the target so we get
; the minimum alignment instead of the preferred alignment.
@byte = global i8 1, align 1
@halfword = global i16 258, align 1
@word = global i32 16909060, align 1
@float = global float 1.0, align 1
@dword = global i64 283686952306183, align 1
@double = global double 1.0, align 1
@pointer = global i8* @byte

; ALL-NOT:       .p2align
; ALL-LABEL: byte:
; ALL:           .byte 1
; ALL:           .size byte, 1

; ALL:           .p2align 1
; ALL-LABEL: halfword:
; ALL:           .2byte 258
; ALL:           .size halfword, 2

; ALL:           .p2align 2
; ALL-LABEL: word:
; ALL:           .4byte 16909060
; ALL:           .size word, 4

; ALL:           .p2align 2
; ALL-LABEL: float:
; ALL:           .4byte 1065353216
; ALL:           .size float, 4

; ALL:           .p2align 3
; ALL-LABEL: dword:
; ALL:           .8byte 283686952306183
; ALL:           .size dword, 8

; ALL:           .p2align 3
; ALL-LABEL: double:
; ALL:           .8byte 4607182418800017408
; ALL:           .size double, 8

; O32:           .p2align 2
; N32:           .p2align 2
; N64:           .p2align 3
; ALL-LABEL: pointer:
; O32:           .4byte byte
; O32:           .size pointer, 4
; N32:           .4byte byte
; N32:           .size pointer, 4
; N64:           .8byte byte
; N64:           .size pointer, 8

@byte_array = global [2 x i8] [i8 1, i8 2], align 1
@halfword_array = global [2 x i16] [i16 1, i16 2], align 1
@word_array = global [2 x i32] [i32 1, i32 2], align 1
@float_array = global [2 x float] [float 1.0, float 2.0], align 1
@dword_array = global [2 x i64] [i64 1, i64 2], align 1
@double_array = global [2 x double] [double 1.0, double 2.0], align 1
@pointer_array = global [2 x i8*] [i8* @byte, i8* @byte]

; ALL-NOT:       .p2align
; ALL-LABEL: byte_array:
; ALL:           .ascii "\001\002"
; ALL:           .size byte_array, 2

; ALL:           .p2align 1
; ALL-LABEL: halfword_array:
; ALL:           .2byte 1
; ALL:           .2byte 2
; ALL:           .size halfword_array, 4

; ALL:           .p2align 2
; ALL-LABEL: word_array:
; ALL:           .4byte 1
; ALL:           .4byte 2
; ALL:           .size word_array, 8

; ALL:           .p2align 2
; ALL-LABEL: float_array:
; ALL:           .4byte 1065353216
; ALL:           .4byte 1073741824
; ALL:           .size float_array, 8

; ALL:           .p2align 3
; ALL-LABEL: dword_array:
; ALL:           .8byte 1
; ALL:           .8byte 2
; ALL:           .size dword_array, 16

; ALL:           .p2align 3
; ALL-LABEL: double_array:
; ALL:           .8byte 4607182418800017408
; ALL:           .8byte 4611686018427387904
; ALL:           .size double_array, 16

; O32:           .p2align 2
; N32:           .p2align 2
; N64:           .p2align 3
; ALL-LABEL: pointer_array:
; O32:           .4byte byte
; O32:           .4byte byte
; O32:           .size pointer_array, 8
; N32:           .4byte byte
; N32:           .4byte byte
; N32:           .size pointer_array, 8
; N64:           .8byte byte
; N64:           .8byte byte
; N64:           .size pointer_array, 16

%mixed = type { i8, double, i16 }
@mixed = global %mixed { i8 1, double 1.0, i16 515 }, align 1

; ALL:           .p2align 3
; ALL-LABEL: mixed:
; ALL:           .byte 1
; ALL:           .space 7
; ALL:           .8byte 4607182418800017408
; ALL:           .2byte 515
; ALL:           .space 6
; ALL:           .size mixed, 24

; Bitfields are not available in LLVM IR so we can't test them here.
