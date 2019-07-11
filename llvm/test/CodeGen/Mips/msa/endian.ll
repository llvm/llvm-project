; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck -check-prefix=BIGENDIAN %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck -check-prefix=LITENDIAN %s

@v16i8 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
@v8i16 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
@v4i32 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>
@v2i64 = global <2 x i64> <i64 0, i64 0>

define void @const_v16i8() nounwind {
  ; LITENDIAN: .byte 0
  ; LITENDIAN: .byte 1
  ; LITENDIAN: .byte 2
  ; LITENDIAN: .byte 3
  ; LITENDIAN: .byte 4
  ; LITENDIAN: .byte 5
  ; LITENDIAN: .byte 6
  ; LITENDIAN: .byte 7
  ; LITENDIAN: .byte 8
  ; LITENDIAN: .byte 9
  ; LITENDIAN: .byte 10
  ; LITENDIAN: .byte 11
  ; LITENDIAN: .byte 12
  ; LITENDIAN: .byte 13
  ; LITENDIAN: .byte 14
  ; LITENDIAN: .byte 15
  ; LITENDIAN: const_v16i8:
  ; BIGENDIAN: .byte 0
  ; BIGENDIAN: .byte 1
  ; BIGENDIAN: .byte 2
  ; BIGENDIAN: .byte 3
  ; BIGENDIAN: .byte 4
  ; BIGENDIAN: .byte 5
  ; BIGENDIAN: .byte 6
  ; BIGENDIAN: .byte 7
  ; BIGENDIAN: .byte 8
  ; BIGENDIAN: .byte 9
  ; BIGENDIAN: .byte 10
  ; BIGENDIAN: .byte 11
  ; BIGENDIAN: .byte 12
  ; BIGENDIAN: .byte 13
  ; BIGENDIAN: .byte 14
  ; BIGENDIAN: .byte 15
  ; BIGENDIAN: const_v16i8:

  store volatile <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, <16 x i8>*@v16i8

  ret void
}

define void @const_v8i16() nounwind {
  ; LITENDIAN: .2byte 0
  ; LITENDIAN: .2byte 1
  ; LITENDIAN: .2byte 2
  ; LITENDIAN: .2byte 3
  ; LITENDIAN: .2byte 4
  ; LITENDIAN: .2byte 5
  ; LITENDIAN: .2byte 6
  ; LITENDIAN: .2byte 7
  ; LITENDIAN: const_v8i16:
  ; BIGENDIAN: .2byte 0
  ; BIGENDIAN: .2byte 1
  ; BIGENDIAN: .2byte 2
  ; BIGENDIAN: .2byte 3
  ; BIGENDIAN: .2byte 4
  ; BIGENDIAN: .2byte 5
  ; BIGENDIAN: .2byte 6
  ; BIGENDIAN: .2byte 7
  ; BIGENDIAN: const_v8i16:

  store volatile <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, <8 x i16>*@v8i16

  ret void
}

define void @const_v4i32() nounwind {
  ; LITENDIAN: .4byte 0
  ; LITENDIAN: .4byte 1
  ; LITENDIAN: .4byte 2
  ; LITENDIAN: .4byte 3
  ; LITENDIAN: const_v4i32:
  ; BIGENDIAN: .4byte 0
  ; BIGENDIAN: .4byte 1
  ; BIGENDIAN: .4byte 2
  ; BIGENDIAN: .4byte 3
  ; BIGENDIAN: const_v4i32:

  store volatile <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32>*@v4i32

  ret void
}

define void @const_v2i64() nounwind {
  ; LITENDIAN: .4byte 1
  ; LITENDIAN: .4byte 0
  ; LITENDIAN: .4byte 2
  ; LITENDIAN: .4byte 0
  ; LITENDIAN: const_v2i64:
  ; BIGENDIAN: .4byte 0
  ; BIGENDIAN: .4byte 1
  ; BIGENDIAN: .4byte 0
  ; BIGENDIAN: .4byte 2
  ; BIGENDIAN: const_v2i64:

  store volatile <2 x i64> <i64 1, i64 2>, <2 x i64>*@v2i64

  ret void
}
