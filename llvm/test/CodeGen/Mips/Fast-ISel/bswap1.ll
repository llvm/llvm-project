; RUN: llc < %s -mtriple=mipsel -mcpu=mips32 -O0 -relocation-model=pic \
; RUN:      -fast-isel-abort=3 | FileCheck %s \
; RUN:      -check-prefix=ALL -check-prefix=32R1
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r2 -O0 -relocation-model=pic \
; RUN:      -fast-isel-abort=3 | FileCheck %s \
; RUN:      -check-prefix=ALL -check-prefix=32R2

@a = global i16 -21829, align 2
@b = global i32 -1430532899, align 4
@a1 = common global i16 0, align 2
@b1 = common global i32 0, align 4

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)

define void @b16() {
  ; ALL-LABEL:  b16:

  ; ALL:            lw    $[[A_ADDR:[0-9]+]], %got(a)($[[GOT_ADDR:[0-9]+]])
  ; ALL:            lhu   $[[A_VAL:[0-9]+]], 0($[[A_ADDR]])

  ; 32R1:           sll   $[[TMP1:[0-9]+]], $[[A_VAL]], 8
  ; 32R1:           srl   $[[TMP2:[0-9]+]], $[[A_VAL]], 8
  ; 32R1:           or    $[[TMP3:[0-9]+]], $[[TMP1]], $[[TMP2]]
  ; 32R1:           andi  $[[TMP4:[0-9]+]], $[[TMP3]], 65535

  ; 32R2:           wsbh  $[[RESULT:[0-9]+]], $[[A_VAL]]

  %1 = load i16, ptr @a, align 2
  %2 = call i16 @llvm.bswap.i16(i16 %1)
  store i16 %2, ptr @a1, align 2
  ret void
}

define void @b32() {
  ; ALL-LABEL:  b32:

  ; ALL:            lw    $[[B_ADDR:[0-9]+]], %got(b)($[[GOT_ADDR:[0-9]+]])
  ; ALL:            lw    $[[B_VAL:[0-9]+]], 0($[[B_ADDR]])

  ; 32R1:           srl   $[[TMP1:[0-9]+]], $[[B_VAL]], 8
  ; 32R1:           srl   $[[TMP2:[0-9]+]], $[[B_VAL]], 24
  ; 32R1:           andi  $[[TMP3:[0-9]+]], $[[TMP1]], 65280
  ; 32R1:           or    $[[TMP4:[0-9]+]], $[[TMP2]], $[[TMP3]]
  ; 32R1:           andi  $[[TMP5:[0-9]+]], $[[B_VAL]], 65280
  ; 32R1:           sll   $[[TMP6:[0-9]+]], $[[TMP5]], 8
  ; 32R1:           sll   $[[TMP7:[0-9]+]], $[[B_VAL]], 24
  ; 32R1:           or    $[[TMP8:[0-9]+]], $[[TMP4]], $[[TMP6]]
  ; 32R1:           or    $[[RESULT:[0-9]+]], $[[TMP7]], $[[TMP8]]

  ; 32R2:           wsbh  $[[TMP:[0-9]+]], $[[B_VAL]]
  ; 32R2:           rotr  $[[RESULT:[0-9]+]], $[[TMP]], 16

  %1 = load i32, ptr @b, align 4
  %2 = call i32 @llvm.bswap.i32(i32 %1)
  store i32 %2, ptr @b1, align 4
  ret void
}
