; RUN: llc  < %s -march=mipsel -mcpu=mips32r2 | FileCheck %s -check-prefix=MIPS32
; RUN: llc  < %s -mtriple=mipsel-mti-linux-gnu -mcpu=mips32r2 -mattr=+micromips | FileCheck %s -check-prefix=MM
; RUN: llc  < %s -march=mips64el -mcpu=mips64r2 | FileCheck %s -check-prefix=MIPS64
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32r2 -mattr=+mips16 | FileCheck %s -check-prefix=MIPS16

define i32 @bswap32(i32 signext %x) nounwind readnone {
entry:
; MIPS32-LABEL: bswap32:
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16

; MM-LABEL: bswap32:
; MM: wsbh $[[R0:[0-9]+]]
; MM: rotr ${{[0-9]+}}, $[[R0]], 16

; MIPS64-LABEL: bswap32:
; MIPS64: wsbh $[[R0:[0-9]+]]
; MIPS64: rotr ${{[0-9]+}}, $[[R0]], 16

; MIPS16-LABEL: bswap32:
; MIPS16-DAG: srl $[[R0:[0-9]+]], $4, 8
; MIPS16-DAG: srl $[[R1:[0-9]+]], $4, 24
; MIPS16-DAG: sll $[[R2:[0-9]+]], $4, 8
; MIPS16-DAG: sll $[[R3:[0-9]+]], $4, 24
; MIPS16-DAG: li  $[[R4:[0-9]+]], 65280
; MIPS16-DAG: and $[[R4]], $[[R0]]
; MIPS16-DAG: or  $[[R1]], $[[R4]]
; MIPS16-DAG: lw  $[[R7:[0-9]+]], $CPI
; MIPS16-DAG: and $[[R7]], $[[R2]]
; MIPS16-DAG: or  $[[R3]], $[[R7]]
; MIPS16-DAG: or  $[[R3]], $[[R1]]

  %or.3 = call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %or.3
}

define i64 @bswap64(i64 signext %x) nounwind readnone {
entry:
; MIPS32-LABEL: bswap64:
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16

; MM-LABEL: bswap64:
; MM: wsbh $[[R0:[0-9]+]]
; MM: rotr ${{[0-9]+}}, $[[R0]], 16
; MM: wsbh $[[R0:[0-9]+]]
; MM: rotr ${{[0-9]+}}, $[[R0]], 16

; MIPS64-LABEL: bswap64:
; MIPS64: dsbh $[[R0:[0-9]+]]
; MIPS64: dshd ${{[0-9]+}}, $[[R0]]

; MIPS16-LABEL: bswap64:
; MIPS16-DAG: srl $[[R0:[0-9]+]], $5, 8
; MIPS16-DAG: srl $[[R1:[0-9]+]], $5, 24
; MIPS16-DAG: sll $[[R2:[0-9]+]], $5, 8
; MIPS16-DAG: sll $[[R3:[0-9]+]], $5, 24
; MIPS16-DAG: li  $[[R4:[0-9]+]], 65280
; MIPS16-DAG: and $[[R0]], $[[R4]]
; MIPS16-DAG: or  $[[R1]], $[[R0]]
; MIPS16-DAG: lw  $[[R7:[0-9]+]], 1f
; MIPS16-DAG: and $[[R2]], $[[R7]]
; MIPS16-DAG: or  $[[R3]], $[[R2]]
; MIPS16-DAG: or  $[[R3]], $[[R1]]
; MIPS16-DAG: srl $[[R0:[0-9]+]], $4, 8
; MIPS16-DAG: srl $[[R1:[0-9]+]], $4, 24
; MIPS16-DAG: sll $[[R2:[0-9]+]], $4, 8
; MIPS16-DAG: sll $[[R3:[0-9]+]], $4, 24
; MIPS16-DAG: li  $[[R4:[0-9]+]], 65280
; MIPS16-DAG: and $[[R0]], $[[R4]]
; MIPS16-DAG: or  $[[R1]], $[[R0]]
; MIPS16-DAG: lw  $[[R7:[0-9]+]], 1f
; MIPS16-DAG: and $[[R2]], $[[R7]]
; MIPS16-DAG: or  $[[R3]], $[[R2]]
; MIPS16-DAG: or  $[[R3]], $[[R1]]

  %or.7 = call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %or.7
}

define <4 x i32> @bswapv4i32(<4 x i32> %x) nounwind readnone {
entry:
; MIPS32-LABEL: bswapv4i32:
; MIPS32-DAG: wsbh $[[R0:[0-9]+]]
; MIPS32-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS32-DAG: wsbh $[[R0:[0-9]+]]
; MIPS32-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS32-DAG: wsbh $[[R0:[0-9]+]]
; MIPS32-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS32-DAG: wsbh $[[R0:[0-9]+]]
; MIPS32-DAG: rotr ${{[0-9]+}}, $[[R0]], 16

; MM-LABEL: bswapv4i32:
; MM-DAG: wsbh $[[R0:[0-9]+]]
; MM-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MM-DAG: wsbh $[[R0:[0-9]+]]
; MM-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MM-DAG: wsbh $[[R0:[0-9]+]]
; MM-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MM-DAG: wsbh $[[R0:[0-9]+]]
; MM-DAG: rotr ${{[0-9]+}}, $[[R0]], 16

; MIPS64-LABEL: bswapv4i32:
; MIPS64-DAG: wsbh $[[R0:[0-9]+]]
; MIPS64-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS64-DAG: wsbh $[[R0:[0-9]+]]
; MIPS64-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS64-DAG: wsbh $[[R0:[0-9]+]]
; MIPS64-DAG: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS64-DAG: wsbh $[[R0:[0-9]+]]
; MIPS64-DAG: rotr ${{[0-9]+}}, $[[R0]], 16

; Don't bother with a MIPS16 version. It's just bswap32 repeated four times and
; would be very long

  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %x)
  ret <4 x i32> %ret
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone

declare i64 @llvm.bswap.i64(i64) nounwind readnone

declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>) nounwind readnone
