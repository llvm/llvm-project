; RUN: llc -mtriple=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-O32
; RUN: llc -mtriple=mipsel -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-O32
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi n32 -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-N32
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi n32 -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-N32
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi n64 -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-N64
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi n64 -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-N64

define float @h() nounwind readnone {
entry:
; PIC-O32: lw  $[[R0:[0-9]+]], %got($CPI0_0)
; PIC-O32: lwc1 $f0, %lo($CPI0_0)($[[R0]])
; STATIC-O32: lui  $[[R0:[0-9]+]], %hi($CPI0_0)
; STATIC-O32: lwc1 $f0, %lo($CPI0_0)($[[R0]])
; PIC-N32: lw  $[[R0:[0-9]+]], %got_page(.LCPI0_0)
; PIC-N32: lwc1 $f0, %got_ofst(.LCPI0_0)($[[R0]])
; STATIC-N32: lui  $[[R0:[0-9]+]], %hi(.LCPI0_0)
; STATIC-N32: lwc1 $f0, %lo(.LCPI0_0)($[[R0]])
; PIC-N64: ld  $[[R0:[0-9]+]], %got_page(.LCPI0_0)
; PIC-N64: lwc1 $f0, %got_ofst(.LCPI0_0)($[[R0]])
; STATIC-N64: lui  $[[R0:[0-9]+]], %highest(.LCPI0_0)
; STATIC-N64: lwc1 $f0, %lo(.LCPI0_0)($[[R0]])
  ret float 0x400B333340000000
}
