; RUN: llc -O0 -march=mipsel -mcpu=mips32r2 -target-abi=o32 < %s -filetype=asm -o - \
; RUN:   | FileCheck -check-prefixes=PTR32,ALL %s
; RUN: llc -O0 -march=mips64el -mcpu=mips64r2 -target-abi=n32 < %s -filetype=asm -o - \
; RUN:   | FileCheck  -check-prefixes=PTR32,ALL %s
; RUN: llc -O0 -march=mips64el -mcpu=mips64r2 -target-abi=n64 < %s -filetype=asm -o - \
; RUN:   | FileCheck -check-prefixes=PTR64,ALL %s


; ALL-LABEL: foo:
; PTR32: lw $[[R0:[0-9]+]]
; PTR32: addiu $[[R1:[0-9]+]], $zero, -4
; PTR32: and $[[R2:[0-9]+]], $[[R0]], $[[R1]]

; PTR64: ld $[[R0:[0-9]+]]
; PTR64: daddiu $[[R1:[0-9]+]], $zero, -4
; PTR64: and $[[R2:[0-9]+]], $[[R0]], $[[R1]]

; ALL: ll ${{[0-9]+}}, 0($[[R2]])

define {i16, i1} @foo(i16** %addr, i16 signext %r, i16 zeroext %new) {
  %ptr = load i16*, i16** %addr
  %res = cmpxchg i16* %ptr, i16 %r, i16 %new seq_cst seq_cst
  ret {i16, i1} %res
}

