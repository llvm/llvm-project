; RUN: llc  < %s -march=mipsel -mcpu=mips32r2 | FileCheck %s -check-prefix=MIPS32
; RUN: llc  < %s -march=mips64el -mcpu=mips64r2 | FileCheck %s -check-prefix=MIPS64

define i32 @bswap32(i32 %x) nounwind readnone {
entry:
; MIPS32: bswap32:
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16
  %or.3 = call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %or.3
}

define i64 @bswap64(i64 %x) nounwind readnone {
entry:
; MIPS64: bswap64:
; MIPS64: dsbh $[[R0:[0-9]+]]
; MIPS64: dshd ${{[0-9]+}}, $[[R0]]
  %or.7 = call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %or.7
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone

declare i64 @llvm.bswap.i64(i64) nounwind readnone

