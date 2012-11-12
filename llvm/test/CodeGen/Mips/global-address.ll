; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-O32
; RUN: llc -march=mipsel -relocation-model=static -mtriple=mipsel-linux-gnu < %s | FileCheck %s -check-prefix=STATIC-O32
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n32 -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-N32
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n32 -relocation-model=static  -mtriple=mipsel-linux-gnu < %s | FileCheck %s -check-prefix=STATIC-N32
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n64 -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-N64
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n64 -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-N64

@s1 = internal unnamed_addr global i32 8, align 4
@g1 = external global i32

define void @foo() nounwind {
entry:
; PIC-O32: lw  $[[R0:[0-9]+]], %got(s1)
; PIC-O32: lw  ${{[0-9]+}}, %lo(s1)($[[R0]])
; PIC-O32: lw  ${{[0-9]+}}, %got(g1)
; STATIC-O32: lui $[[R1:[0-9]+]], %hi(s1)
; STATIC-O32: lw  ${{[0-9]+}}, %lo(s1)($[[R1]])
; STATIC-O32: lui $[[R2:[0-9]+]], %hi(g1)
; STATIC-O32: lw  ${{[0-9]+}}, %lo(g1)($[[R2]])

; PIC-N32: lw  $[[R0:[0-9]+]], %got_page(s1)
; PIC-N32: lw  ${{[0-9]+}}, %got_ofst(s1)($[[R0]])
; PIC-N32: lw  ${{[0-9]+}}, %got_disp(g1)
; STATIC-N32: lui $[[R1:[0-9]+]], %hi(s1)
; STATIC-N32: lw  ${{[0-9]+}}, %lo(s1)($[[R1]])
; STATIC-N32: lui $[[R2:[0-9]+]], %hi(g1)
; STATIC-N32: lw  ${{[0-9]+}}, %lo(g1)($[[R2]])

; PIC-N64: ld  $[[R0:[0-9]+]], %got_page(s1)
; PIC-N64: lw  ${{[0-9]+}}, %got_ofst(s1)($[[R0]])
; PIC-N64: ld  ${{[0-9]+}}, %got_disp(g1)
; STATIC-N64: ld  $[[R1:[0-9]+]], %got_page(s1)
; STATIC-N64: lw  ${{[0-9]+}}, %got_ofst(s1)($[[R1]])
; STATIC-N64: ld  ${{[0-9]+}}, %got_disp(g1)

  %0 = load i32* @s1, align 4
  tail call void @foo1(i32 %0) nounwind
  %1 = load i32* @g1, align 4
  store i32 %1, i32* @s1, align 4
  %add = add nsw i32 %1, 2
  store i32 %add, i32* @g1, align 4
  ret void
}

declare void @foo1(i32)

