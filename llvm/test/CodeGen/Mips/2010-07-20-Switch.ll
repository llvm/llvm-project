; RUN: llc < %s -march=mips -relocation-model=static | \
; RUN: FileCheck %s -check-prefix=STATIC-O32 
; RUN: llc < %s -march=mips -relocation-model=pic | \
; RUN: FileCheck %s -check-prefix=PIC-O32 
; RUN: llc < %s -march=mips64 -relocation-model=pic -mcpu=mips4 | \
; RUN:     FileCheck %s -check-prefix=N64
; RUN: llc < %s -march=mips64 -relocation-model=static -mcpu=mips4 | \
; RUN:     FileCheck %s -check-prefix=N64
; RUN: llc < %s -march=mips64 -relocation-model=pic -mcpu=mips64 | \
; RUN:     FileCheck %s -check-prefix=N64
; RUN: llc < %s -march=mips64 -relocation-model=static -mcpu=mips64 | \
; RUN:     FileCheck %s -check-prefix=N64

define i32 @main() nounwind readnone {
entry:
  %x = alloca i32, align 4                        ; <i32*> [#uses=2]
  store volatile i32 2, i32* %x, align 4
  %0 = load volatile i32, i32* %x, align 4             ; <i32> [#uses=1]
; STATIC-O32: sll $[[R0:[0-9]+]], ${{[0-9]+}}, 2
; STATIC-O32: lui $[[R1:[0-9]+]], %hi($JTI0_0)
; STATIC-O32: addu $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; STATIC-O32: lw $[[R3:[0-9]+]], %lo($JTI0_0)($[[R2]])
; PIC-O32: sll $[[R0:[0-9]+]], ${{[0-9]+}}, 2
; PIC-O32: lw $[[R1:[0-9]+]], %got($JTI0_0)
; PIC-O32: addu $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; PIC-O32: lw $[[R4:[0-9]+]], %lo($JTI0_0)($[[R2]])
; PIC-O32: addu $[[R5:[0-9]+]], $[[R4:[0-9]+]]
; PIC-O32: jr  $[[R5]]
; N64: dsll $[[R0:[0-9]+]], ${{[0-9]+}}, 3
; N64: ld $[[R1:[0-9]+]], %got_page(.LJTI0_0)
; N64: daddu $[[R2:[0-9]+]], $[[R0:[0-9]+]], $[[R1]]
; N64: ld $[[R4:[0-9]+]], %got_ofst(.LJTI0_0)($[[R2]])
; N64: daddu $[[R5:[0-9]+]], $[[R4:[0-9]+]]
; N64: jr  $[[R5]]
  switch i32 %0, label %bb4 [
    i32 0, label %bb5
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
  ]

bb1:                                              ; preds = %entry
  ret i32 2

bb2:                                              ; preds = %entry
  ret i32 0

bb3:                                              ; preds = %entry
  ret i32 3

bb4:                                              ; preds = %entry
  ret i32 4

bb5:                                              ; preds = %entry
  ret i32 1
}

; STATIC-O32: .p2align  2
; STATIC-O32: $JTI0_0:
; STATIC-O32: .4byte
; STATIC-O32: .4byte
; STATIC-O32: .4byte
; STATIC-O32: .4byte
; PIC-O32: .p2align  2
; PIC-O32: $JTI0_0:
; PIC-O32: .gpword
; PIC-O32: .gpword
; PIC-O32: .gpword 
; PIC-O32: .gpword 
; N64: .p2align  3
; N64: .LJTI0_0:
; N64: .gpdword
; N64: .gpdword
; N64: .gpdword 
; N64: .gpdword 

