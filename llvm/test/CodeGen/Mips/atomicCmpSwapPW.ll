; RUN: llc -O0 -march=mipsel -mcpu=mips32r2 -target-abi=o32 < %s -filetype=asm -o - \
; RUN:   | FileCheck -check-prefixes=PTR32,ALL %s
; RUN: llc -O0 -march=mips64el -mcpu=mips64r2 -target-abi=n32 < %s -filetype=asm -o - \
; RUN:   | FileCheck  -check-prefixes=PTR32,ALL %s
; RUN: llc -O0 -march=mips64el -mcpu=mips64r2 -target-abi=n64 < %s -filetype=asm -o - \
; RUN:   | FileCheck -check-prefixes=PTR64,ALL %s

; PTR32: lw $[[R0:[0-9]+]]
; PTR64: ld $[[R0:[0-9]+]]

; ALL: ll ${{[0-9]+}}, 0($[[R0]])

define {i16, i1} @foo(i16* %addr, i16 signext %r, i16 zeroext %new) {
  %res = cmpxchg i16* %addr, i16 %r, i16 %new seq_cst seq_cst
  ret {i16, i1} %res
}

