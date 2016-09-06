; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-O32
; RUN: llc -march=mipsel -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-O32
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi n32 -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-N32
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi n32 -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-N32
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi n64 -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-N64
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi n64 -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-N64
; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32 -mattr=+mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-MIPS16-1
; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32 -mattr=+mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-MIPS16-2

@reg = common global i8* null, align 4

define i8* @dummy(i8* %x) nounwind readnone noinline {
entry:
  ret i8* %x
}

; PIC-O32: lw  $[[R0:[0-9]+]], %got($tmp[[T0:[0-9]+]])
; PIC-O32: addiu ${{[0-9]+}}, $[[R0]], %lo($tmp[[T0]])
; PIC-O32: lw  $[[R1:[0-9]+]], %got($tmp[[T1:[0-9]+]])
; PIC-O32: addiu ${{[0-9]+}}, $[[R1]], %lo($tmp[[T1]])
; STATIC-O32: lui  $[[R2:[0-9]+]], %hi($tmp[[T2:[0-9]+]])
; STATIC-O32: addiu ${{[0-9]+}}, $[[R2]], %lo($tmp[[T2]])
; STATIC-O32: lui   $[[R3:[0-9]+]], %hi($tmp[[T3:[0-9]+]])
; STATIC-O32: addiu ${{[0-9]+}}, $[[R3]], %lo($tmp[[T3]])
; PIC-N32: lw  $[[R0:[0-9]+]], %got_page(.Ltmp[[T0:[0-9]+]])
; PIC-N32: addiu ${{[0-9]+}}, $[[R0]], %got_ofst(.Ltmp[[T0]])
; PIC-N32: lw  $[[R1:[0-9]+]], %got_page(.Ltmp[[T1:[0-9]+]])
; PIC-N32: addiu ${{[0-9]+}}, $[[R1]], %got_ofst(.Ltmp[[T1]])
; STATIC-N32: lui  $[[R2:[0-9]+]], %hi(.Ltmp[[T2:[0-9]+]])
; STATIC-N32: addiu ${{[0-9]+}}, $[[R2]], %lo(.Ltmp[[T2]])
; STATIC-N32: lui   $[[R3:[0-9]+]], %hi(.Ltmp[[T3:[0-9]+]])
; STATIC-N32: addiu ${{[0-9]+}}, $[[R3]], %lo(.Ltmp[[T3]])
; PIC-N64: ld  $[[R0:[0-9]+]], %got_page(.Ltmp[[T0:[0-9]+]])
; PIC-N64: daddiu ${{[0-9]+}}, $[[R0]], %got_ofst(.Ltmp[[T0]])
; PIC-N64: ld  $[[R1:[0-9]+]], %got_page(.Ltmp[[T1:[0-9]+]])
; PIC-N64: daddiu ${{[0-9]+}}, $[[R1]], %got_ofst(.Ltmp[[T1]])
; STATIC-N64: ld  $[[R2:[0-9]+]], %got_page(.Ltmp[[T2:[0-9]+]])
; STATIC-N64: daddiu ${{[0-9]+}}, $[[R2]], %got_ofst(.Ltmp[[T2]])
; STATIC-N64: ld  $[[R3:[0-9]+]], %got_page(.Ltmp[[T3:[0-9]+]])
; STATIC-N64: daddiu ${{[0-9]+}}, $[[R3]], %got_ofst(.Ltmp[[T3]])
; STATIC-MIPS16-1: .ent	f
; STATIC-MIPS16-2: .ent	f
; STATIC-MIPS16-1: li  $[[R1_16:[0-9]+]], %hi($tmp[[TI_16:[0-9]+]])
; STATIC-MIPS16-1: sll ${{[0-9]+}},  $[[R1_16]], 16
; STATIC-MIPS16-2: li  ${{[0-9]+}}, %lo($tmp{{[0-9]+}})
; STATIC-MIPS16-1: jal	dummy
; STATIC-MIPS16-2: jal	dummy

define void @f() nounwind {
entry:
  %call = tail call i8* @dummy(i8* blockaddress(@f, %baz))
  indirectbr i8* %call, [label %baz, label %foo]

foo:                                              ; preds = %foo, %entry
  store i8* blockaddress(@f, %foo), i8** @reg, align 4
  br label %foo

baz:                                              ; preds = %entry
  store i8* null, i8** @reg, align 4
  ret void
}
