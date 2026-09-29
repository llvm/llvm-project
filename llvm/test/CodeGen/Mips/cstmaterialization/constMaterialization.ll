; RUN: llc -mtriple=mips < %s | FileCheck %s -check-prefixes=ALL,MIPS
; RUN: llc -mtriple=mips < %s -mattr=+micromips | FileCheck %s -check-prefixes=ALL,MM

; Test the patterns used for constant materialization.

; Constants generated using li16
define i32 @Li16LowBoundary() {
entry:
  ; ALL-LABEL: Li16LowBoundary:
  ; MIPS:     addiu	$2, $zero, -1
  ; MM:       li16	$2, -1
  ; ALL-NOT:  lui
  ; ALL-NOT:  ori
  ; MIPS-NOT: li16
  ; MM-NOT:   addiu

  ret i32 -1
}

define i32 @Li16HighBoundary() {
entry:
  ; ALL-LABEL: Li16HighBoundary:
  ; MIPS:     addiu	$2, $zero, 126
  ; MM:       li16	$2, 126
  ; ALL-NOT:  lui
  ; ALL-NOT:  ori
  ; MM-NOT:   addiu
  ; MIPS-NOT: li16

  ret i32 126
}

; Constants generated using addiu
define i32 @AddiuLowBoundary() {
entry:
  ; ALL-LABEL: AddiuLowBoundary:
  ; ALL:      addiu	$2, $zero, -32768
  ; ALL-NOT:  lui
  ; ALL-NOT:  ori
  ; ALL-NOT:  li16

  ret i32 -32768
}

define i32 @AddiuZero() {
entry:
  ; ALL-LABEL: AddiuZero:
  ; MIPS:     addiu	$2, $zero, 0
  ; MM:       li16	$2, 0
  ; ALL-NOT:  lui
  ; ALL-NOT:  ori
  ; MIPS-NOT: li16
  ; MM-NOT:   addiu

  ret i32 0
}

define i32 @AddiuHighBoundary() {
entry:
  ; ALL-LABEL: AddiuHighBoundary:
  ; ALL:     addiu	$2, $zero, 32767
  ; ALL-NOT: lui
  ; ALL-NOT: ori
  ; ALL-NOT: li16

  ret i32 32767
}

; Constants generated using ori
define i32 @OriLowBoundary() {
entry:
  ; ALL-LABEL: OriLowBoundary:
  ; ALL:     ori	$2, $zero, 32768
  ; ALL-NOT: addiu
  ; ALL-NOT: lui
  ; ALL-NOT: li16

  ret i32 32768
}

define i32 @OriHighBoundary() {
entry:
  ; ALL-LABEL: OriHighBoundary:
  ; ALL:     ori	$2, $zero, 65535
  ; ALL-NOT: addiu
  ; ALL-NOT: lui
  ; ALL-NOT: li16

  ret i32 65535
}

; Constants generated using lui
define i32 @LuiPositive() {
entry:
  ; ALL-LABEL: LuiPositive:
  ; ALL:     lui	$2, 1
  ; ALL-NOT: addiu
  ; ALL-NOT: ori
  ; ALL-NOT: li16

  ret i32 65536
}

define i32 @LuiNegative() {
entry:
  ; ALL-LABEL: LuiNegative:
  ; ALL:     lui	$2, 65535
  ; ALL-NOT: addiu
  ; ALL-NOT: ori
  ; ALL-NOT: li16

  ret i32 -65536
}

; Constants generated using a combination of lui and ori
define i32 @LuiWithLowBitsSet() {
entry:
  ; ALL-LABEL: LuiWithLowBitsSet:
  ; ALL:     lui	$1, 1
  ; ALL:     ori	$2, $1, 1
  ; ALL-NOT: addiu
  ; ALL-NOT: li16

  ret i32 65537
}

define i32 @BelowAddiuLowBoundary() {
entry:
  ; ALL-LABEL: BelowAddiuLowBoundary:
  ; ALL:     lui	$1, 65535
  ; ALL:     ori	$2, $1, 32767
  ; ALL-NOT: addiu
  ; ALL-NOT: li16

  ret i32 -32769
}
